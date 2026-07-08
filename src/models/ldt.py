from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack
from einops.layers.torch import Rearrange
import lightning as L
from main import instantiate_from_config
from src.modules.transformer.attention import Transformer, ContinuousPositionBias




def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class LatentActionQuantization(L.LightningModule):
    def __init__(
        self,
        lossconfig,
        vq_config,
        dim,
        patch_size,
        spatial_depth,
        temporal_depth,
        temporal_patch_size,
        ckpt_path = None,
        image_size=[112,96],
        dim_head = 64,
        heads = 8,
        channels = 1,
        attn_dropout = 0.,
        ff_dropout = 0.,
        code_seq_len = 1,
        learning_rate=None,
        sche_type=None,
        resume_lr = None,
        max_it = None,
        breakdown_type = False,
        wp = 0,
        wp0 = 0.005, #initial lr ratio at the begging of lr warm up
        wpe = 0.01, #final lr ratio at the end of training
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.code_seq_len = code_seq_len
        self.image_size = image_size
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        self.loss = instantiate_from_config(lossconfig)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        
        transformer_with_action_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
            has_cross_attn = True,
            dim_context = dim,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)

        self.breakdown_type = breakdown_type
        self.vq = instantiate_from_config(vq_config)
            
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.max_it = max_it
        self.learning_rate = learning_rate
        self.sche_type = sche_type
        self.resume_lr = resume_lr
        ## for scheduler
        self.wp = wp
        self.wp0 = wp0
        self.wpe = wpe
        
        # Enable manual optimization for multiple optimizers (Generator and Discriminator)
        # This is required for GAN training where we need to separately optimize
        # the generator and discriminator networks
        self.automatic_optimization = False
       
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)
    
    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace('module.', '') if 'module.' in k else k: v for k, v in pt.items()}
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]

        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        #tokens shape[B,2t,h,w,d]

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        tokens = rearrange(tokens, 'b (n t) h w d -> (b t h w) n d', n=2)

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t h w) n d -> b (n t) h w d', b = b, h = h, w = w, n=2)

        t = tokens.shape[1] // 2
        first_tokens = tokens[:, :t] #[B,t,h,w,d]
        last_tokens = tokens[:, t:] #[B,t,h,w,d]

        return first_tokens, last_tokens

        

    def decode(
        self,
        tokens,
        actions,
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])


        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        # actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape, context=actions)
        

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        rest_frames_tokens = tokens

        recon_video = self.to_pixels(rest_frames_tokens)

        return recon_video
    

    def forward(
        self,
        pre,
        post,
        step = 0,
        mask = None,
        return_recons_only = False,
        return_only_codebook_ids = False,
    ):

        b, c, f, *image_dims, device = *pre.shape, pre.device
        assert not exists(mask) or mask.shape[-1] == f

        pre_tokens = self.to_patch_emb(pre)
        post_tokens = self.to_patch_emb(post)
        tokens = torch.cat((pre_tokens, post_tokens), dim = 1) #(B,2xt,h,w,d)

        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        first_tokens = rearrange(first_tokens, 'b t h w d -> (b t) (h w) d')
        last_tokens = rearrange(last_tokens, 'b t h w d -> (b t) (h w) d')

        if self.breakdown_type:
            (tokens, emb_loss, indices), loss_breakdown = self.vq(first_tokens, last_tokens, return_loss_breakdown=True)
        else:
            tokens, perplexity, codebook_usage, indices = self.vq(first_tokens, last_tokens, codebook_training_only = False)
            num_unique_indices = indices.unique().size(0)
        
            if ((step % 10 == 0 and step < 100)  or (step % 100 == 0 and step < 1000) or (step % 500 == 0 and step < 5000)) and step != 0:
                print(f"update codebook {step}")
                self.vq.replace_unused_codebooks(tokens.shape[0])

        if return_only_codebook_ids:
            return indices
        
        # if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
        #     action_h = int(math.sqrt(self.code_seq_len))
        #     action_w = int(math.sqrt(self.code_seq_len))
        # elif self.code_seq_len == 2:
        #     action_h = 2
        #     action_w = 1
        # else:
        #     ## error
        #     print("code_seq_len should be square number or defined as 2")
        #     return
        
        # tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = pre_tokens.detach() # + tokens
        recon_video = self.decode(concat_tokens, tokens)

        if return_recons_only:
            return recon_video

        # returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        # video = post
        return recon_video, emb_loss, loss_breakdown if self.breakdown_type else recon_video
    

    @torch.no_grad()
    def inference(
        self,
        pre,
        indices,
    ):

        pre_tokens = self.to_patch_emb(pre)
        tokens = self.vq.indices2quantized(indices, indices.shape[0])

        concat_tokens = pre_tokens.detach() # + tokens

        recon_video = self.decode(concat_tokens, tokens)

        return recon_video


    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            opt_gen, opt_disc = self.optimizers()
            for opt_gen_param_group, opt_disc_param_group in zip(opt_gen.param_groups, opt_disc.param_groups):
                opt_gen_param_group["lr"] = self.resume_lr
                opt_disc_param_group["lr"] = self.resume_lr

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        
        g_it = self.trainer.global_step
        pre = batch['pre_token'][:,:,1:]
        post = batch['post_token'][:,:,1:]


        if self.breakdown_type:
            xrec, eloss,  loss_break = self(pre, post, step = g_it)
        else:
            xrec = self(pre, post, step = g_it)

        x = post

        ###Adjuts the learning rate
        if self.sche_type is not None and self.resume_lr is None:
           
            if self.max_it is None:
                iters_train = len(self.trainer.train_dataloader) ## get the total iterations in a epoch
                max_it = self.trainer.max_epochs * iters_train
                wp_it = self.wp * iters_train
            else:
                max_it = max_it
                wp_it = self.wp_iter
            self.lr_annealing(self.learning_rate, g_it, wp_it, max_it, wp0=self.wp0, wpe=self.wpe)

        opt_gen, opt_disc = self.optimizers()
        # scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        # original VQGAN first optimizes G, then D. We first optimize D then G, following traditional GAN
        
        # optimize generator
        if self.breakdown_type:
                aeloss, log_dict_ae = self.loss(eloss, loss_break, x, xrec, 0, self.global_step,
                                    last_layer=self.get_last_layer(), split="train")
        else:
            aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_gen.zero_grad()
        self.manual_backward(aeloss)
        opt_gen.step()
        
        # optimize discriminator
        if self.breakdown_type:
                discloss, log_dict_disc = self.loss(eloss, loss_break, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        else:
            discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        
    # def validation_step(self, batch, batch_idx, suffix=""):
        
    #     pre = batch['pre_token'][:,:,1:]
    #     post = batch['post_token'][:,:,1:]

    #     x_rec = self(pre, post, step = 0)
    #     x = post

    #     aeloss, log_dict_ae = self.loss(x, x_rec, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), split="val"+ suffix)

    #     discloss, log_dict_disc = self.loss(x, x_rec, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val" + suffix)

    #     self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    #     self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    #     return self.log_dict

    def get_last_layer(self):
        """
        Get the last layer of the decoder for discriminator loss calculation
        """
        return self.to_pixels[-2].weight

    def configure_optimizers(self):
        """
        Configure optimizers for generator and discriminator
        Following the same pattern as vae_lfq.py
        """
        lr = self.learning_rate
        
        # Generator parameters (all model parameters except discriminator)
        gen_params = []
        for module in [self.to_patch_emb, self.enc_spatial_transformer, self.enc_temporal_transformer, 
                       self.vq, self.dec_spatial_transformer, self.to_pixels]:
            gen_params.extend(list(module.parameters()))
        
        opt_gen = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.9))
        
        # Discriminator parameters
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        
        return [opt_gen, opt_disc], []
    
    def lr_annealing(self, peak_lr, cur_it, wp_it, max_it, wp0=0.005, wpe=0.01):
        """
        Learning rate annealing for multiple optimizers
        Modified from VAR, following vae_lfq.py implementation
        """
        wp_it = round(wp_it)
        if cur_it < wp_it:
            cur_lr = wp0 + (1-wp0) * cur_it / wp_it
        else:
            pasd = (cur_it - wp_it) / (max_it-1 - wp_it)   # [0, 1]
            rest = 1 - pasd     # [1, 0]
            if self.sche_type == "lin0":
                T = 0.05; max_rest = 1-T
                if pasd < T: cur_lr = 1
                else: cur_lr = wpe + (1-wpe) * rest / max_rest
            elif self.sche_type == "cos":
                cur_lr = wpe + (1-wpe) * (0.5 + 0.5 * math.cos(math.pi * pasd))
            elif self.sche_type == "constant":
                cur_lr = 1.0
            else:  # default to linear decay
                T = 0.05; max_rest = 1-T
                if pasd < T: cur_lr = 1
                else: cur_lr = wpe + (1-wpe) * rest / max_rest

        cur_lr *= peak_lr
        
        # Apply to both optimizers
        opt_gen, opt_disc = self.optimizers()
        ### adjust Generator Learning Rate
        for param_group in opt_gen.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)

        ### adjust Discriminator Learning Rate
        for param_group in opt_disc.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)