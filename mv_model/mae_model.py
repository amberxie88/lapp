import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions.dirichlet import Dirichlet
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

# adapted from Multi-MAE https://github.com/EPFL-VILAB/MultiMAE/blob/11167059599e563a0edb9cb48e1dc8ab45ad4b92/multimae/input_adapters.py
class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.
    :param num_channels: Number of input channels of the image/feature map
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 patch_size_full: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 learnable_pos_emb: bool = False,
                 image_size: Union[int, Tuple[int]] = 256):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size_full) * (self.image_size[1] // patch_size_full)

        self.P_H = max(1, self.patch_size_full[0])
        self.P_W = max(1, self.patch_size_full[1])

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.
        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.P_H
        w_posemb = self.image_size[1] // self.P_W
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.
        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (P_H*P_W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x

class MAEModel(BaseModel):
	"""
	Custom Multiview-MAE Implementation
	"""

	def __init__(self, name, device, lr, loss, dropout, bias, weight_decay, token_dim,
				 obs_encoder_id, viewpoints, viewpoint_embedding, mask_ratio, 
				 encoder_n_layer, encoder_n_head, decoder_n_layer,decoder_n_head,
				 reconstruct_cls, scheduler, schedule_every_step,
				 encoder_embed_dim, decoder_embed_dim):
		super(MAEModel, self).__init__()
		self.device = device

		# mask sampling
		self.mask_ratio = mask_ratio

		# viewpoint embeddings
		self.num_viewpoints = len(viewpoints)
		self.viewpoint_embedding = viewpoint_embedding
		if viewpoint_embedding == True:
			self.viewpoint_embedding_lst = nn.ParameterList()
			self.decoder_viewpoint_embedding_lst = nn.ParameterList()
			for _ in range(self.num_viewpoints):
				vp_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, encoder_embed_dim)), std=0.02))
				self.viewpoint_embedding_lst.append(vp_embedding)
				vp_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, decoder_embed_dim)), std=0.02))
				self.decoder_viewpoint_embedding_lst.append(vp_embedding)
		self.mask_token = nn.Parameter(init.normal_(torch.empty((1, 1, decoder_embed_dim)), std=0.02))
		self.cls_token = nn.Parameter(init.normal_(torch.empty((1, 1, encoder_embed_dim)), std=0.02))

		self.obs_encoder_id = obs_encoder_id
		if self.obs_encoder_id == "multimae_patch":
			self.patch_size = patch_size = 16
			self.num_patches_one_direction = 14 
			self.vision_model = PatchedInputAdapter(3, patch_size, token_dim, sincos_pos_emb=True, learnable_pos_emb=False, image_size=256).to(self.device)
		else:
			raise NotImplementedError

		# MAE Encoder and Decoder
		self.token_dim = token_dim
		self.mae_encoder = nn.ModuleDict(dict(
			drop = nn.Dropout(dropout),
			h = nn.ModuleList([Block(encoder_embed_dim, encoder_n_head, bias, dropout) for _ in range(encoder_n_layer)]),
			norm = LayerNorm(encoder_embed_dim, bias=bias), 
		)).to(self.device)

		self.mae_decoder = nn.ModuleDict(dict(
			pred = nn.Linear(decoder_embed_dim, token_dim),
			h = nn.ModuleList([Block(decoder_embed_dim, decoder_n_head, bias, dropout) for _ in range(decoder_n_layer)]),
			norm = LayerNorm(decoder_embed_dim, bias=bias),
			embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
		)).to(self.device)

		# positional embedding for decoder
		sincos_pos_emb = True
		if sincos_pos_emb:
		    pos_emb = build_2d_sincos_posemb(h=self.num_patches_one_direction, w=self.num_patches_one_direction, embed_dim=decoder_embed_dim)
		    pos_emb = nn.Parameter(pos_emb, requires_grad=False)
		    pos_emb = F.interpolate(pos_emb, size=(self.num_patches_one_direction, self.num_patches_one_direction), mode='bicubic', align_corners=False).to(self.device)
		    self.decoder_pos_embed = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
		else:
		    self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
		    trunc_normal_(self.pos_emb, std=0.02)

		self.loss = loss

		# networks to save and optimize
		self.networks = ['mae_encoder', 'mae_decoder', 'mask_token', 'cls_token', 'vision_model']
		if viewpoint_embedding is True:
			self.networks.append('viewpoint_embedding_lst')
			self.networks.append('decoder_viewpoint_embedding_lst')
		self.optims = [self.configure_optims(lr, weight_decay)]
		self.schedulers = self.configure_schedulers(scheduler, schedule_every_step)
		self.to(self.device)

	def forward(self, data):
		return self.forward_verbose(data)[0]

	def forward_verbose(self, data):
		meta = dict()

		# get observation embeddings
		B = data['actual_obs'].shape[0]
		viewpoint_tokens = []
		vit_obs = data['vit_obs'].to(self.device)
		data['vit_obs'] = vit_obs
		obs_token = self.vision_model(vit_obs)
		viewpoint_tokens.append(obs_token)
		
		if 'extra_vit_obs' in data.keys() and data['extra_vit_obs'].shape[-1] > 0:
			all_extra_vit_obs = data['extra_vit_obs'].to(self.device)
			data['extra_vit_obs'] = all_extra_vit_obs
			for extra_vit_obs_id in range(all_extra_vit_obs.shape[1]):
				extra_obs_token = self.vision_model(all_extra_vit_obs[:, extra_vit_obs_id, :, :])
				viewpoint_tokens.append(extra_obs_token)
		n_views = len(viewpoint_tokens)
		
		# add viewpoint embeddings
		for i in range(len(viewpoint_tokens)):
			viewpoint_tokens[i] = viewpoint_tokens[i] + self.get_viewpoint_embedding(i)

		# masking
		viewpoint_token_len = [vp_token.shape[1] for vp_token in viewpoint_tokens]
		num_tokens_prefix_sum = utils.prefix_sum(viewpoint_token_len)
		num_mask = round(self.mask_ratio * num_tokens_prefix_sum[-1])
		num_mask_per_viewpoint = [round(tl * self.mask_ratio) for tl in viewpoint_token_len]
		num_mask_per_viewpoint = torch.tensor(num_mask_per_viewpoint, device=self.device).unsqueeze(0).repeat(B, 1)

		# sample masks for each view
		viewpoint_masks = []
		for i in range(len(viewpoint_tokens)):
			viewpoint_token_len = viewpoint_tokens[i].shape[1]
			noise = torch.rand(B, viewpoint_token_len, device=self.device)
			shuffled_indices = torch.argsort(noise, dim=1)
			mask = torch.where(shuffled_indices < (num_mask_per_viewpoint)[:, i].unsqueeze(1), 1, 0)
			all_indices = torch.argsort(mask)
			viewpoint_masks.append(mask)

		final_mask = torch.cat(viewpoint_masks, dim=1)
		indices_shuffle = torch.argsort(-final_mask)
		mask_indices = indices_shuffle[:, :num_mask]
		no_mask_indices = indices_shuffle[:, num_mask:]
		indices_restore = torch.argsort(indices_shuffle, dim=1)

		# concat all masked sequence
		unmasked_seq = torch.cat(viewpoint_tokens, dim=1)
		masked_seq = torch.gather(unmasked_seq, dim=1, index=no_mask_indices.unsqueeze(-1).repeat(1, 1, unmasked_seq.shape[2]))
		cls_tokens = self.cls_token.expand(masked_seq.shape[0], -1, -1)
		masked_seq = torch.cat((cls_tokens, masked_seq), dim=1)

		# MAE Encoder
		feats = self.forward_mae_encoder(masked_seq)

		# MAE Decoder
		feats = self.mae_decoder.embed(feats)
		mask_tokens = self.mask_token.repeat(feats.shape[0], mask_indices.shape[1], 1)
		mask_nomask_tokens = torch.cat([mask_tokens, feats[:, 1:, :]], dim=1) # no CLS
		featurized_seq = torch.gather(mask_nomask_tokens, dim=1, index=indices_restore.unsqueeze(-1).repeat(1, 1, mask_nomask_tokens.shape[2]))  # unshuffle

		# positional embedding is critical
		featurized_seq = featurized_seq + self.decoder_pos_embed.repeat(1,n_views,1)
		if self.viewpoint_embedding == True:
			viewpoint_embeds = [self.get_decoder_viewpoint_embedding(i).repeat(1,196,1) for i in range(n_views)]
			viewpoint_embeds = torch.cat(viewpoint_embeds, dim=1)
			featurized_seq += viewpoint_embeds

		featurized_seq = torch.cat((feats[:,:1,:],featurized_seq), dim=1) # append CLS again
		featurized_seq = self.forward_mae_decoder(featurized_seq)
		featurized_seq = featurized_seq[:, 1:, :] # remove CLS

		meta['masks'] = final_mask.detach().unsqueeze(-1)
		meta['pixel_pred'] = featurized_seq
		meta['pixel_pred_w_mask'] = featurized_seq * (meta['masks'])
		return featurized_seq, meta

	def forward_mae_encoder(self, feats):
		for block in self.mae_encoder.h:
			feats = block(feats)
		feats = self.mae_encoder.norm(feats)
		return feats

	def forward_mae_decoder(self, featurized_seq):
		for block in self.mae_decoder.h:
			featurized_seq = block(featurized_seq)
		featurized_seq = self.mae_decoder.norm(featurized_seq)
		featurized_seq = self.mae_decoder.pred(featurized_seq) 
		return featurized_seq

	def get_viewpoint_embedding(self, idx):
		if self.viewpoint_embedding:
			return self.viewpoint_embedding_lst[idx]
		else:
			return 0.0

	def get_decoder_viewpoint_embedding(self, idx):
		if self.viewpoint_embedding:
			return self.decoder_viewpoint_embedding_lst[idx]
		else:
			return 0.0

	def unpatchify(self, patchified_pixel_values):
		# https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/vit_mae/modeling_vit_mae.py
		batch_size = patchified_pixel_values.shape[0]
		num_channels = 3
		patchified_pixel_values = patchified_pixel_values.reshape(
		    batch_size,
		    self.num_patches_one_direction,
		    self.num_patches_one_direction,
		    self.patch_size,
		    self.patch_size,
		    num_channels,
		)
		patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
		pixel_values = patchified_pixel_values.reshape(
		    batch_size,
		    num_channels,
		    self.num_patches_one_direction * self.patch_size,
		    self.num_patches_one_direction * self.patch_size,
		)
		return pixel_values

	def patchify(self, pixel_values):
		# https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/vit_mae/modeling_vit_mae.py
		patch_size, num_channels = self.patch_size, 3
		# sanity checks
		if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
		    raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
		if pixel_values.shape[1] != num_channels:
		    raise ValueError(
		        "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
		    )

		# patchify
		batch_size = pixel_values.shape[0]
		patchified_pixel_values = pixel_values.reshape(
		    batch_size, num_channels, self.num_patches_one_direction, patch_size, self.num_patches_one_direction, patch_size
		)
		patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
		patchified_pixel_values = patchified_pixel_values.reshape(
		    batch_size, self.num_patches_one_direction * self.num_patches_one_direction, patch_size**2 * num_channels
		)
		return patchified_pixel_values

	def get_loss(self, data, pred, meta):
		metrics = dict()

		assert pred.shape[1] % 196 == 0 
		differences = []
		for idx in range(pred.shape[1] // 196):
			if idx == 0:
				img_patch = data['vit_obs']
			else:
				img_patch = data['extra_vit_obs'][:, idx-1, :, :, :]
			img_patch = self.patchify(img_patch)
			pred_patch = pred[:, 196*idx:196*(idx+1), :]
			difference = (img_patch - pred_patch) * meta['masks'][:, 196*idx:196*(idx+1), :] / meta['masks'].sum() * torch.prod(torch.tensor(meta['masks'].shape))
			differences.append(difference)

		loss = 0
		if self.loss == "l2_mean":
			for difference in differences:
				loss += difference.square().mean()
			loss /= len(differences)
		else:
			raise NotImplementedError
		metrics['mae_loss'] = loss.item()

		metrics['loss'] = loss.item()
		return loss, metrics

	def update(self, data, step, dataset=None):
		pred, meta = self.forward_verbose(data)
		loss, metrics = self.get_loss(data, pred, meta)
		for optim in self.optims:
			optim.zero_grad()
		loss.backward()
		for optim in self.optims:
			optim.step()
		metrics['schedule_step'] = 0
		if self.schedule_every_step(step):
			metrics['schedule_step'] = 1 
			for scheduler in self.schedulers:
				if scheduler is not None:
					scheduler.step()
		if self.schedulers[0] is not None:
			metrics['lr'] = self.schedulers[0].get_last_lr()[0]
		return metrics

	def get_eval_metrics(self, model_out, data):
		pred, meta = model_out
		with torch.no_grad():
			_, metrics = self.get_loss(data, pred, meta)

		return metrics 

	def configure_optims(self, lr, weight_decay):
		# separate out all parameters to those that will and won't experience regularizing weight decay
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (nn.Linear, )
		blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding)
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
		for name, param in self.named_parameters():
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
				other[key].cpu()
				self.__dict__['_modules'][key].to(self.device)
			elif key in self.__dict__['_parameters'].keys():
				self.__dict__['_parameters'][key].data.copy_(other[key].data)
				other[key].cpu()
			else:
				if isinstance(other[key], list):
					for i in range(len(other[key])):
						self.__dict__[key][i].data.copy_(other[key][i].data)
				else:
					self.__dict__[key].data.copy_(other[key].data)

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