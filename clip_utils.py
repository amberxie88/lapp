import transformers

def load_model(encoder):
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    elif encoder in ["clip16", "clip16_finetune"]:
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    elif encoder == "clip16_scratch":
        cfg = transformers.CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch16")
        model = transformers.CLIPTextModel(cfg)
    elif encoder == "clip14":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    elif encoder == "clip14_scratch":
        cfg = transformers.CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
        model = transformers.CLIPTextModel(cfg)
    elif encoder == "clip_scratch":
        cfg = transformers.CLIPTextConfig.from_pretrained("openai/clip-vit-base-patch32")
        model = transformers.CLIPTextModel(cfg)
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model

def load_vision_model(encoder):
    if encoder == "clip16":
        model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16") 
    elif encoder in ["clip16_scratch", "clip16_finetune"]:
        cfg = transformers.CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
        model = transformers.CLIPVisionModel(cfg)
    elif encoder == "clip14":
        model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14") 
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    return model

def load_tokenizer(encoder):
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder in ["clip", "clip_scratch"]:
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    elif encoder in ["clip16", "clip16_scratch", "clip16_finetune"]:
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
    elif encoder in ["clip14", "clip14_scratch"]:
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer