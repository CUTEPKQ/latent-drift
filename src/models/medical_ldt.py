"""
Refer to 
https://github.com/FoundationVision/LlamaGen
https://github.com/FoundationVision/VAR
"""

import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L
import inspect
from einops import rearrange, repeat
import numpy as np
from main import instantiate_from_config


from peft import LoraConfig, get_peft_model,TaskType
from src.modules.transformer.action_gpt import sample_from_logits

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Net2NetTransformer(L.LightningModule):
    def __init__(self,
                encoder_config,
                transformer_config,
                ldt_config,
                ckpt_path=None,
                ignore_keys=[],
                pkeep=1.0,
                learning_rate=None,
                use_w = False,
                use_lora = False,
                lora_config = None,
                token_factorization=False,
                weight_decay=1e-2,
                resume_lr = None,
                wp = 0,
                wp0 = 0.005, #initial lr ratio at the begging of lr warm up
                wpe = 0.01, #final lr ratio at the end of training
                twde = 0
                 ):
        super().__init__()

        self.encoder = instantiate_from_config(config=encoder_config)

        self.transformer = instantiate_from_config(config=transformer_config)
        self.use_w = use_w

        self.init_first_stage_from_ckpt(ldt_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
  
        self.pkeep = pkeep

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.token_factorization = token_factorization
        self.resume_lr = resume_lr

        ## for scheduler
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        self.twde = twde or weight_decay
        
 
        ## for loar
        if use_lora:
            if lora_config is None:
                lora_config = {
                    "r": 16,
                    # "lora_alpha": 16,
                    # "lora_dropout": 0.05,   # 0.05~0.1
                    # "bias": "none",
                    # "target_modules": ["wqkv", "wo","w1", "w2", "w3"],
                    "target_modules": ["wqkv"],
                    "task_type": TaskType.FEATURE_EXTRACTION
                }
            peft_config = LoraConfig(**lora_config)
            self.transformer = get_peft_model(self.transformer, peft_config)
            self.transformer.print_trainable_parameters()


    def state_dict(self, *kwargs, destination=None, prefix='', keep_vars=False):
        return {k: v for k, v in super().state_dict(*kwargs, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        self.first_stage_model.requires_grad_(False)


    def forward(self, pre, time_diff, cond_drop_prob=0, handcraft_f=None,**kwargs):
        feature = self.encoder(pre)[:,:,1:] # (b c t h w) t:24->23
        feature = rearrange(feature, 'b c t h w -> b (t h w) c')
        logits = self.transformer(feature, idx_cls=time_diff, cond_drop_prob=cond_drop_prob, handcraft_f=handcraft_f, **kwargs)
        return logits
    
    def forward_with_cfg(self, pre, time_diff, cond_scale=1.0):
        feature = self.encoder(pre)[:,:,1:] # (b c t h w) t:24->23
        feature = rearrange(feature, 'b c t h w -> b (t h w) c')
        logits = self.transformer.forward_with_cond_scale(
                                    feature=feature, 
                                    idx_cls=time_diff, 
                                    cond_scale=cond_scale)
        return logits

    @torch.no_grad()
    def decode_to_video(self, pre, action_indices):
        return self.first_stage_model.inference(pre, action_indices)
  
    def shared_step(self, batch, batch_idx, cond_drop_prob=0):

        pre = batch['pre_token']
        post = batch['post_token']
        time_diff = batch['time_diff']
        if self.transformer.config.use_handcraft:
            handcraft_f = batch['handcraft_f']
        else:
            handcraft_f = None

        target_token = self.first_stage_model(pre[:,:,1:], post[:,:,1:], return_only_codebook_ids=True).long()
    
        if len(target_token.shape) == 2:
            target_token = rearrange(target_token, '(b t) l -> b (t l)', b=pre.shape[0])
            # 传 targets 让 VARActionGPT 等 AR transformer 能做 teacher forcing；
            # 单 codebook feedforward transformer (如 ActionGPT) 形参里有 targets= 但不用，安全。
            logits = self(pre, time_diff=time_diff, cond_drop_prob=cond_drop_prob, handcraft_f=handcraft_f, targets=target_token)
        elif len(target_token.shape) ==3:
            target_token = rearrange(target_token, '(b t) k l -> b k (t l)', b=pre.shape[0])
            logits = self(pre, time_diff=time_diff, cond_drop_prob=cond_drop_prob, handcraft_f=handcraft_f,targets=target_token)

      
        B = logits.shape[0]

        # Check for nan/inf in logits before sampling
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"Warning: Found nan or inf in logits at step {batch_idx}")
            print(f"Logits stats - min: {logits.min().item()}, max: {logits.max().item()}, "
                  f"nan count: {torch.isnan(logits).sum().item()}, inf count: {torch.isinf(logits).sum().item()}")
        

        gen_token = logits.argmax(dim=-1)
        accuracy = (gen_token == target_token).float().mean()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_token.reshape(-1))
        return loss, accuracy


    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt = self.optimizers()
            for opt_param_group in opt.param_groups:
                opt_param_group["lr"] = self.resume_lr

    def training_step(self, batch, batch_idx):
        ## control lr and wd
        ## --------------------------------------------
        iters_train = len(self.trainer.train_dataloader) ## get the total iterations in a epoch
        g_it = self.trainer.global_step
        max_it = self.trainer.max_epochs * iters_train
        wp_it = self.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = self.lr_wd_annealing(self.learning_rate, self.weight_decay, self.twde, g_it, wp_it, max_it, wp0=self.wp0, wpe=self.wpe)
        ## --------------------------------------------
        B = batch['pre_token'].shape[0]
        cond_drop_prob = self.transformer.cond_drop_prob 
        loss, acc = self.shared_step(batch, batch_idx, cond_drop_prob=cond_drop_prob)
        # Log learning rate and weight decay for monitoring
        self.log("train/lr", max_tlr, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
        self.log("train/wd", max_twd, prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False)
       
        
        # Log overall metrics
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("train/acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
      
        return loss

#     def validation_step(self, batch, batch_idx):
#  ## --------------------------------------------
#         B = batch['pre_token'].shape[0]
#         # default cond_drop_prob = 0 
#         loss, acc = self.shared_step(batch, batch_idx)

#         # Log overall metrics
#         self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)
#         self.log("val/acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=B)

#         return loss

    def configure_optimizers(self):
        """
        Following NanoGPT, since we adopt the Llama-Like framework for AutoRegressive Visual Generation
        """

        # Collect parameters from both encoder and transformer
        param_dict = {}
        for pn, p in self.encoder.named_parameters():
            param_dict[f'encoder.{pn}'] = p
        for pn, p in self.transformer.named_parameters():
            param_dict[f'transformer.{pn}'] = p
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused_available = False
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95), **extra_args)

        return optimizer
    
    def lr_wd_annealing(self, peak_lr, wd, wd_end, cur_it, wp_it, max_it, wp0=0.005, wpe=0.001):
        """
        Modified from VAR
        """
        wp_it = round(wp_it)
        if cur_it < wp_it:
            cur_lr = wp0 + (1-wp0) * cur_it / wp_it
        else:
            pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
            rest = 1 - pasd     # [1, 0]
            ## using linear decay by default
            T = 0.05; max_rest = 1-T
            if pasd < T: cur_lr = 1
            else: cur_lr = wpe + (1-wpe) * rest / max_rest

        cur_lr *= peak_lr
        pasd = cur_it / (max_it-1)
        cur_wd = wd_end + (wd - wd_end) * (0.5 + 0.5 * math.cos(math.pi * pasd))
    
        inf = 1e6
        min_lr, max_lr = inf, -1
        min_wd, max_wd = inf, -1
        for param_group in self.optimizers().param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)    # 'lr_sc' could be assigned
            max_lr = max(max_lr, param_group['lr'])
            min_lr = min(min_lr, param_group['lr'])
            
            param_group['weight_decay'] = cur_wd * param_group.get('wd_sc', 1)
            max_wd = max(max_wd, param_group['weight_decay'])
            if param_group['weight_decay'] > 0:
                min_wd = min(min_wd, param_group['weight_decay'])

        if min_lr == inf: min_lr = -1
        if min_wd == inf: min_wd = -1
        return min_lr, max_lr, min_wd, max_wd