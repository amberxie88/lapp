import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import inspect
import re
import utils
from clip_utils import load_model, load_tokenizer, load_vision_model
from transformer_utils import build_2d_sincos_posemb, pair, CrossBlock, Block, LayerNorm
from typing import Union, Tuple, Optional
from einops import rearrange, repeat

from laco_model.base_model import BaseModel

class LACOModel(BaseModel):
	"""
	Transformer-based Collision Detection module that takes in language, image, and state.
	"""

	def __init__(self, name, device, lr, state_hidden_dim, net_hidden_dim,
				 obs_encoder_id, threshold, encoder, model_max_length, freeze_language_encoder,
				 modality_embedding, patch_size, token_dim, use_mv, dropout, n_layer, bias, 
				 n_head, weight_decay, pred_coll_loss, alpha_coll_loss, attention_feat, n_obs_attn_layer, 
				 use_transformer_pos_emb, scheduler, schedule_every_step, state_dim,
				 normalize_state, mv_train_status):
		super(LACOModel, self).__init__()
		self.device = device
		self.obs_encoder_id = obs_encoder_id

		self.threshold = threshold # past threshold indicates collision

		self.modality_embedding = modality_embedding
		if modality_embedding == True:
			self.obs_type_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, token_dim)), std=0.02))
			self.lang_type_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, token_dim)), std=0.02))
			self.state_type_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, token_dim)), std=0.02))
			
		self.state_dim = state_dim
		if isinstance(state_hidden_dim, int):
			raise NotImplementedError
		else:
			model = [nn.Linear(state_dim, state_hidden_dim[0]), nn.ReLU()]
			for i in range(0, len(state_hidden_dim) - 1):
				model += [nn.Linear(state_hidden_dim[i], state_hidden_dim[i+1]), nn.ReLU()] 
			self.state_encoder = nn.Sequential(*model).to(self.device) 
			self.state_linear = nn.Linear(state_hidden_dim[-1], token_dim).to(self.device)

		if self.obs_encoder_id == "mae":
			pass
		elif self.obs_encoder_id in ["clip16", "clip14", "clip16_scratch", "clip16_finetune"]:
			encoder = self.obs_encoder_id
			self.vision_model = load_vision_model(self.obs_encoder_id).to(self.device)
		else:
			raise NotImplementedError

		# transformer
		self.transformer = nn.ModuleDict(dict(
			drop = nn.Dropout(dropout),
			h = nn.ModuleList([Block(token_dim, n_head, bias, dropout) for _ in range(n_layer)]),
			cross = None,
			ln_f = LayerNorm(token_dim, bias=bias),
		)).to(self.device)
		self.attention_feat = attention_feat
		self.token_dim = token_dim
		if self.token_dim != 768:
			self.obs_linear = nn.Linear(768, token_dim)

		# additional observation attention layers
		self.use_transformer_pos_emb = use_transformer_pos_emb
		if self.use_transformer_pos_emb:
		    pos_emb = build_2d_sincos_posemb(h=14, w=14, embed_dim=token_dim)
		    pos_emb = nn.Parameter(pos_emb, requires_grad=False)
		    pos_emb = F.interpolate(pos_emb, size=(14, 14), mode='bicubic', align_corners=False).to(self.device)
		    self.transformer_pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
		self.use_mv = use_mv 
		self.mv_train_status = mv_train_status
		assert not self.use_mv or self.use_mv != (n_obs_attn_layer > 0), "use_mv and n_obs_attn_layer must be mutually exclusive"
		self.n_obs_attn_layer = n_obs_attn_layer
		if self.n_obs_attn_layer > 0:
			self.obs_attn = nn.ModuleList([Block(token_dim, n_head, bias, dropout) for _ in range(self.n_obs_attn_layer)]).to(self.device)

		# auxiliary losses
		self.pred_coll_loss, self.alpha_coll_loss = pred_coll_loss, alpha_coll_loss
		
		# prediction head
		model = [nn.Linear(token_dim, net_hidden_dim[0]), nn.ReLU()]
		for i in range(0, len(net_hidden_dim) - 1):
			model += [nn.Linear(net_hidden_dim[i], net_hidden_dim[i+1]), nn.ReLU()]
		model += [nn.Linear(net_hidden_dim[-1], 1+int(pred_coll_loss), bias=False), nn.Sigmoid()] 
		self.pred_head = nn.Sequential(*model).to(self.device) 

		# language model 
		self.tokenizer = load_tokenizer(encoder)
		self.tokenizer.model_max_length = model_max_length
		model = load_model(encoder)
		self.model = model.to(self.device)
		if "clip16" in encoder or encoder == "clip":
			self.proj_lang_encoder = nn.Linear(512, token_dim).to(self.device)
		elif "clip14" in encoder:
			self.proj_lang_encoder = nn.Linear(768, token_dim).to(self.device)
		else:
			raise NotImplementedError

		# networks to save and optimize
		self.networks = ['state_encoder', 'state_linear', 
						'pred_head',
						'proj_lang_encoder', 'transformer']
		if self.obs_encoder_id == "mae":
			self.networks.append('obs_tokenizer')
		if self.n_obs_attn_layer > 0:
			self.networks.append('obs_attn')
		if self.modality_embedding:
			self.networks.extend(['obs_type_embedding', 'lang_type_embedding', 'state_type_embedding'])
		if self.token_dim != 768:
			self.networks.append('obs_linear')
		if self.obs_encoder_id in ["clip16_scratch", "clip16_finetune"]:
			self.networks.append('vision_model')
		if self.use_mv and self.mv_train_status != "frozen":
			self.networks.append('mv_model')
			# not sure if need this or not
		self.optims = [self.configure_optims(lr, weight_decay)]
		self.schedulers = self.configure_schedulers(scheduler, schedule_every_step)
		self.normalize_state = normalize_state
		if self.normalize_state:
			self.state_mean = torch.tensor([-3.44671526, -0.01341378, -2.8631923 ,  1.54004452, -3.14917949, 0.25634965, -2.54531042], device=self.device)

		# train/freeze language encoder
		self.freeze_language_encoder = freeze_language_encoder
		if freeze_language_encoder:
			utils.freeze_params(self.model)
		else: 
			self.networks['model'] = self.model 
			self.optims.append(torch.optim.Adam(self.model.parameters(),
                        lr=lr, betas=[0.9, 0.999]))

		# losses
		self.loss = nn.BCELoss()

		self.to(self.device)

	def add_mv_model(self, mv_model):
		self.mv_model = mv_model

	def forward(self, data):
		return self.forward_verbose(data)[0]

	def forward_verbose(self, data):
		meta = dict()
		obs, states, langs = data['init_obs'], data['states'], data['language']
		obs = obs.to(self.device)
		# language tokens
		tokens = self.tokenizer(langs, padding="max_length")["input_ids"]
		tokens = torch.tensor(tokens).to(self.device)
		if self.freeze_language_encoder:
			with torch.no_grad():
				lang_embedding = self.model(tokens).last_hidden_state
		else:
			lang_embedding = self.model(tokens).last_hidden_state
		lang_token = self.proj_lang_encoder(lang_embedding) + self.get_type_embedding('lang')

		# state token
		states = states.to(self.device).float()
		if self.normalize_state:
			states = states - self.state_mean
		states_enc = self.state_encoder(states)
		states_token = self.state_linear(states_enc).unsqueeze(1) + self.get_type_embedding('state')

		# observation tokens
		all_extra_tokens = []
		with torch.no_grad():
			if self.obs_encoder_id == "mae":
				with torch.no_grad():
					# get observation embeddings
					B = data['actual_obs'].shape[0]
					viewpoint_tokens = []
					vit_obs = data['vit_obs'].to(self.device)
					data['vit_obs'] = vit_obs
					obs_token = self.mv_model.vision_model(vit_obs)
					viewpoint_tokens.append(obs_token)
				
					if 'extra_vit_obs' in data.keys() and data['extra_vit_obs'].shape[-1] > 0:
						all_extra_vit_obs = data['extra_vit_obs'].to(self.device)
						data['extra_vit_obs'] = all_extra_vit_obs
						for extra_vit_obs_id in range(all_extra_vit_obs.shape[1]):
							extra_obs_token = self.mv_model.vision_model(all_extra_vit_obs[:, extra_vit_obs_id, :, :])
							viewpoint_tokens.append(extra_obs_token)

					# concat all masked sequence
					unmasked_seq = torch.cat(viewpoint_tokens, dim=1)
					cls_tokens = self.mv_model.cls_token.expand(unmasked_seq.shape[0], -1, -1)
					obs_token = torch.cat((cls_tokens, unmasked_seq), dim=1)
			elif self.obs_encoder_id in ["clip16", "clip14"]:
				vit_obs = data['vit_obs'].to(self.device)
				obs_token = self.vision_model(vit_obs)['last_hidden_state']
		if self.obs_encoder_id in ["clip16_scratch", "clip16_finetune"]:
			vit_obs = data['vit_obs'].to(self.device)
			obs_token = self.vision_model(vit_obs)['last_hidden_state']
		if len(all_extra_tokens) > 0:
			extra_obs_token = torch.cat(all_extra_tokens, dim=1)
			all_obs_token = torch.cat([extra_obs_token, obs_token], dim=1)
		else:
			all_obs_token = obs_token
		featurized_obs_tokens = self.forward_obs_attn(all_obs_token)
		featurized_obs_tokens = featurized_obs_tokens + self.get_type_embedding('obs')
		
		if self.use_transformer_pos_emb:
			if self.obs_encoder_id == "mae":
				featurized_obs_tokens[:, 1:197, :] += self.transformer_pos_emb
				featurized_obs_tokens[:, 197:, :] += self.transformer_pos_emb
			else:
				raise NotImplementedError
		# concatenate all tokens
		context = torch.cat([featurized_obs_tokens, lang_token], dim=1)
		
		sequence = torch.cat([context, states_token], dim=1)
		x = self.transformer.drop(sequence)
		for block in self.transformer.h:
			x = block(x)
		if self.attention_feat == "average":
			x = x.mean(dim=1)
		elif self.attention_feat == "last":
			x = x[:, -1, :]
		else:
			raise NotImplementedError
		x = x.squeeze(1)
		x = self.transformer.ln_f(x)
		pred = self.pred_head(x)
		return pred, meta

	def forward_obs_attn(self, obs_tokens):
		if self.use_mv:
			if self.mv_train_status == "frozen":
				with torch.no_grad():
					return self.mv_model.forward_mae_encoder(obs_tokens)
			else: 
				return self.mv_model.forward_mae_encoder(obs_tokens)
		elif self.n_obs_attn_layer > 0:
			for block in self.obs_attn:
			    obs_tokens = block(obs_tokens)
			return obs_tokens
		if self.token_dim != 768:
			obs_tokens = self.obs_linear.forward(obs_tokens)
			return obs_tokens
		else:
			return obs_tokens


	def get_type_embedding(self, name):
		if self.modality_embedding:
			return {
				'obs': self.obs_type_embedding,
				'lang': self.lang_type_embedding,
				'state': self.state_type_embedding,
			}[name]
		else:
			return 0.0

	def get_metrics(self, pred, data):
		metrics = self.get_eval_metrics(pred, data)
		# remove the following keys to remove unnecessary logging
		metrics.pop('num_collisions')
		metrics.pop('num_nocollision')
		metrics.pop('num_collision_change')
		metrics.pop('num_batch')
		return metrics

	def get_eval_metrics(self, model_out, data):
		if self.pred_coll_loss:
			pred, pred_coll = model_out[:, 0].unsqueeze(1), model_out[:, 1].unsqueeze(1)
		else:
			pred = model_out
		b = data['collisions'].shape[0]

		collisions = data['collisions']
		if len(data['all_collisions'].shape) > 1:
			all_collisions = data['all_collisions'].sum(dim=1)
		else:
			all_collisions = data['all_collisions']

		any_collision = all_collisions.clamp(0, 1).float()
		collision_change = 1 - ((all_collisions > 0) == data['collisions']).float()
		percent_collision_change = collision_change.float().mean()

		n_collisions = torch.sum(collisions)
		percent_collision = n_collisions / b
		avg_predict = pred.mean()
		pred_collision = (pred > self.threshold).float().squeeze(1)
		assert pred_collision.shape == collisions.shape
		overall_accuracy = (pred_collision == collisions).float()
		n_collisions = torch.clamp(n_collisions, 1, b-1)
		collision_accuracy = ((collisions * overall_accuracy).sum() / n_collisions)
		nocollision_accuracy = ((1-collisions) * overall_accuracy).sum() / (1 - collisions).sum()
		nocollision_accuracy = nocollision_accuracy
		n_collision_change = torch.clamp(collision_change.sum(), 1, b-1)
		collision_change_accuracy = ((collision_change * overall_accuracy).sum() / n_collision_change)
		with torch.no_grad():
			loss = self.loss(pred.squeeze(1), collisions)
			if self.pred_coll_loss:
				pred_coll_loss = self.loss(pred_coll.squeeze(1), any_collision)
			
		metrics = dict(percent_collision=percent_collision, avg_predict=avg_predict,
					overall_accuracy=overall_accuracy.mean(), collision_accuracy=collision_accuracy,
					nocollision_accuracy=nocollision_accuracy, 
					collision_change_accuracy=collision_change_accuracy,
					percent_collision_change=percent_collision_change,
					num_collisions=n_collisions, num_collision_change=collision_change.sum(),
					num_nocollision=b-n_collisions, num_batch=b, loss=loss)
		if self.pred_coll_loss:
			overall_pred_coll_accuracy = ((pred_coll.squeeze(1) > 0.5).float() == any_collision).float()
			metrics['predcoll_collision_accuracy'] = ((any_collision * overall_pred_coll_accuracy).sum() / any_collision.sum())
			metrics['predcoll_no_collision_accuracy'] = (((1-any_collision) * overall_pred_coll_accuracy).sum() / (1-any_collision).sum())
			metrics['predcoll_overall_accuracy'] = overall_pred_coll_accuracy.mean()
			metrics['predcoll_loss'] = pred_coll_loss
		
		return metrics 

	def update(self, data, step, optimize=True):
		metrics = dict()
		collisions = data['collisions']
		collisions = collisions.to(self.device)
		data['collisions'] = collisions
		data['all_collisions'] = data['all_collisions'].to(self.device)
		model_out, meta = self.forward_verbose(data)
		if self.pred_coll_loss:
			pred, coll_change = model_out[:, 0].unsqueeze(1), model_out[:, 1].unsqueeze(1)
		else:
			pred = model_out

		loss = self.loss(pred.squeeze(1), collisions)
		if self.pred_coll_loss:
			if len(data['all_collisions'].shape) > 1:
				all_collisions = data['all_collisions'].sum(dim=1)
			else:
				all_collisions = data['all_collisions']

			any_collision = torch.clamp(all_collisions, 0, 1).float()
			pred_coll_loss = self.loss(coll_change.squeeze(1), any_collision)
			loss += self.alpha_coll_loss * pred_coll_loss
		if optimize:
			for optim in self.optims:
				optim.zero_grad()
			if self.use_mv and self.mv_train_status != "frozen":
				for optim in self.mv_model.optims:
					optim.zero_grad()
			loss.backward()
			for optim in self.optims:
				optim.step()
			if self.use_mv and self.mv_train_status != "frozen":
				for optim in self.mv_model.optims:
					optim.step()
		else:
			loss = loss.detach()

		metrics.update(self.get_metrics(model_out, data))
		metrics['final_loss'] = loss

		metrics['schedule_step'] = 0
		if self.schedule_every_step(step):
			metrics['schedule_step'] = 1 
			for scheduler in self.schedulers:
				if scheduler is not None:
					scheduler.step()
		if self.schedulers[0] is not None:
			metrics['lr'] = self.schedulers[0].get_last_lr()[0]

		return metrics

	def configure_optims(self, lr, weight_decay):
		# separate out all parameters to those that will and won't experience regularizing weight decay
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear, )
		blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			relevant_module = False
			for mod in self.networks:
				if mod in mn:
					relevant_module = True
					break
			if relevant_module is False:
				continue
			for pn, p in m.named_parameters():
			    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
			    # random note: because named_modules and named_parameters are recursive
			    # we will see the same tensors p many many times. but doing it this way
			    # allows us to know which parent module any tensor p belongs to...
			    # print(fpn, end="|")
			    if pn.endswith('bias'):
			        # all biases will not be decayed
			        no_decay.add(fpn)
			    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
			        # weights of whitelist modules will be weight decayed
			        decay.add(fpn)
			    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
			        # weights of blacklist modules will NOT be weight decayed
			        no_decay.add(fpn)

		# if not already found. Not sure why the above iterates over modules instead of parameters
		for name, _ in self.named_parameters():
			relevant = False
			for net in self.networks:
				if net in name:
					relevant = True
			if not relevant:
				continue
			if name in decay or name in no_decay:
				continue
			no_decay.add(name)

		# create the pytorch optimizer object
		param_dict = {pn: p for pn, p in self.named_parameters()}
		optim_groups = [
		    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
		    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]
		# new PyTorch nightly has a new 'fused' option for AdamW that is much faster
		use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		print(f"using fused AdamW: {use_fused}")
		extra_args = dict(fused=True) if use_fused else dict()
		optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=[0.9, 0.999], **extra_args)
		return optimizer

	def configure_schedulers(self, scheduler, schedule_every_step):
		self.schedule_every_step = utils.Every(schedule_every_step)
		schedulers = []
		for optim in self.optims:
			match = re.match(r'cosine\((.+),(.+)\)', scheduler)
			if match:
				T_max, eta_min = [float(g) for g in match.groups()] 
				sched = lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=eta_min)
			else:
				print("No scheduler")
				sched = None
			schedulers.append(sched)
		return schedulers

	def init_from(self, other):
		# copy parameters over
		for key in self.networks:
			if key in self.__dict__['_modules'].keys():
				utils.hard_update_params(other[key], self.__dict__['_modules'][key])
				self.__dict__['_modules'][key].to(self.device)
			elif key in self.__dict__['_parameters'].keys():
				self.__dict__['_parameters'][key].data.copy_(other[key].data)
				self.__dict__['_parameters'][key].to(self.device)
			elif key == 'mv_model':
				print("not initing mv model yet")
				continue
			else:
				self.__dict__[key].data.copy_(other[key].data)
				self.__dict__[key].to(self.device)
			other[key].cpu()

	def get_model(self):
		model_dict = dict()
		for key in self.networks:
			if key in self.__dict__['_modules'].keys():
				model_dict[key] = self.__dict__['_modules'][key]
			elif key in self.__dict__['_parameters'].keys():
				model_dict[key] = self.__dict__['_parameters'][key]
			else:
				model_dict[key] = self.__dict__[key]
		return model_dict