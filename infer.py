#!/usr/bin/env python3
"""
GPT model sampling test script.
Tests the GPT model's sampling: given a pretoken, generate the posttoken,
then decode it into a video using the VQ model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import cv2
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange
# Add src to the import path
import sys
sys.path.append('src')

from src.models.medical_ldt import Net2NetTransformer
from src.data.ct_video import CTVideoDataset
from main import DataModuleFromConfig


def save_middle_slice_comparison(original, reconstructed, output_path):
    """
    Save a comparison image of the middle slice.

    Args:
        original_tensor: original tensor (1, D, H, W)
        reconstructed_tensor: reconstructed tensor (1, D, H, W)
        output_path: output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid_slice = len(original) // 2

    orig_slice = original[mid_slice]
    recon_slice = reconstructed[mid_slice]
    # Normalize to [0, 1]
    orig_slice = orig_slice/255.0
    recon_slice = recon_slice/255.0


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(orig_slice, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(recon_slice, cmap='gray')
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')

    # Difference map
    diff = np.abs(orig_slice - recon_slice)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Middle slice comparison saved to: {output_path}")


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    """Sample the next token from logits."""
    B, L, V = logits.shape
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k and top_k > 0 or top_p and top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = torch.topk(probs, k=1, dim=-1)
    else:
        probs = probs.view(B * L, V)              # [B*L, V]
        x = torch.multinomial(probs, num_samples=1)  # [B*L, 1]
        x = x.view(B, L)                          # [B, L]

    return x

def np_2_nii(np_data, output_nii_path, rotate90_hw=True):
    """
    Convert a numpy array back into a nii file.

    Args:
        np_data: numpy array, shape (1, 93, 112, 96), value range [-1, 1]
        output_nii_path: output nii file path
        target_shape: target nii shape, default (112, 96, 93) -> (H, W, D)
        rotate90_hw: whether to undo the 90-degree rotation, default True
    """
    import nibabel as nib

    # 1. Convert to tensor
    t = torch.from_numpy(np_data).float()  # (1, 93, 112, 96)
    t = t.unsqueeze(0)  # (1, 1, 93, 112, 96) = (1, 1, D, H, W)

    # 2. Undo the rotation (if one was applied earlier)
    if rotate90_hw:
        t = torch.rot90(t, k=-1, dims=(3, 4))  # Rotate the H-W plane back

    # 4. Reorder dimensions (1, 1, D, H, W) -> (H, W, D)
    vol = t.squeeze(0).squeeze(0).permute(1, 2, 0).numpy()  # (H, W, D) = (112, 96, 93)

    # 5. Undo normalization: [-1, 1] -> [-1000, 1000]
    vol = vol * 1000  # [-1, 1] -> [-1000, 1000]

    # 6. Add the offset back: [-1000, 1000] -> [24, 2024] (undo the -1024 step)
    vol = vol + 1024.0

    # 7. Create the nii file
    nii_img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(nii_img, output_nii_path)


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_config(config_path):
    """Load the config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config):
    """Load a trained model."""
    print(f"Loading model from: {checkpoint_path}")

    model_config = config['model']['init_args']
    model = Net2NetTransformer(**model_config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
 
    model.eval()
    
    print("Model loaded successfully!")
    return model

def generate_samples_from_gpt(model, dataset, output_dir, device='cpu',  
                             temperature=1.0, top_k=50, top_p=0.9, max_new_tokens=None, save_nii=False):
    """
    Generate sampling results using the GPT model.

    Args:
        model: trained Net2NetTransformer model
        dataset: CTGenDataset dataset
        output_dir: output directory
        device: compute device
        max_samples: maximum number of samples to process
        temperature: sampling temperature
        top_k: top-k sampling parameter
        top_p: nucleus sampling parameter
        max_new_tokens: max number of tokens to generate; if None, derived from the pretoken length
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(dataset)

    
    print(f"Processing {total_samples} samples for GPT sampling...")
    
    model = model.to(device)
    model.eval()
    
    results = []

    np_dir = os.path.join(output_dir, 'np')
    os.makedirs(np_dir, exist_ok=True)

    nii_dir = os.path.join(output_dir, 'nii')
    if save_nii:
        os.makedirs(nii_dir, exist_ok=True)

    video_dir = os.path.join(output_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device == 'cuda' else False)

    sample_idx = 0

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):

        pretoken = batch_data['pre_token'].to(device)  # shape: [seq_len]
        posttoken_gt = batch_data['post_token'].to(device)
        time_diff = batch_data['time_diff']

        if model.transformer.config.use_handcraft:
            handcraft_f = batch_data['handcraft_f'].to(device)
        else:
            handcraft_f = None

        with torch.no_grad():
            # # Generate the sequence

            generated_logits = model(pretoken,time_diff, handcraft_f=handcraft_f)
            # # # generated_logits = model.forward_with_cfg(pretoken, time_diff, cond_scale=2.8)
    
            gen_token = sample_from_logits(
                generated_logits, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p, 
                sample_logits=True  
            )  #
            # # gen_token = gen_token.argmax(dim=-1)

            gen_token = rearrange(gen_token, 'b (t l) -> (b t) l',l = 20)

            # feature = model.encoder(pretoken)[:,:,1:] # (b c t h w) t:24->23
            # feature = rearrange(feature, 'b c t h w -> b (t h w) c')
            
            # gen_token = model.transformer.generate(feature,idx_cls=time_diff)
            
            pre_first_slice = pretoken[:,:,0:1]
            post_first_slice_gt = posttoken_gt[:,:,0:1]

            pretoken = pretoken[:,:,1:]
            posttoken_gt = posttoken_gt[:,:,1:]


          
            generated_video = model.decode_to_video(pretoken,gen_token)
            generated_video = torch.clamp(generated_video, -1, 1)

            batch_size_actual = posttoken_gt.shape[0]
            for i in range(batch_size_actual):

                source_video = pretoken[i].cpu().numpy()
                gt_video = posttoken_gt[i].cpu().numpy()

                gen_np = generated_video[i].cpu().numpy()

                gt_s = gt_video-source_video
                gen_s = gen_np-source_video

                # np.save(os.path.join(np_dir, f"rec_{sample_idx:04d}.npy"), rec_np)
                np.save(os.path.join(np_dir, f"gen_{sample_idx:04d}.npy"), gen_np)
                np.save(os.path.join(np_dir, f"gt_{sample_idx:04d}.npy"), gt_video)
                np.save(os.path.join(np_dir, f"gt-source_{sample_idx:04d}.npy"),gt_s)
                np.save(os.path.join(np_dir, f"gen-source_{sample_idx:04d}.npy"),gen_s)


                sample_name = f"sample_{sample_idx:04d}"

                if save_nii:
                    nii_sub_dir = os.path.join(nii_dir, sample_name)
                    os.makedirs(nii_sub_dir, exist_ok=True)

                    nii_pre_path = os.path.join(nii_sub_dir, "cur.nii.gz")
                    if not os.path.exists(nii_pre_path):
                        # Prepend the pre first slice to rebuild a 93-frame volume
                        src_full = np.concatenate([pre_first_slice[i].cpu().numpy(), source_video], axis=1)
                        np_2_nii(src_full, nii_pre_path, rotate90_hw=True)

                    nii_post_path = os.path.join(nii_sub_dir, f"fut_{(time_diff[i]+1)*6}month.nii.gz")
                    if not os.path.exists(nii_post_path):
                        gen_full = np.concatenate([post_first_slice_gt[i].cpu().numpy(), gen_np], axis=1)
                        np_2_nii(gen_full, nii_post_path, rotate90_hw=True)

                    nii_gt_path = os.path.join(nii_sub_dir, f"gt_{(time_diff[i]+1)*6}month.nii.gz")
                    if not os.path.exists(nii_gt_path):
                        gt_full = np.concatenate([post_first_slice_gt[i].cpu().numpy(), gt_video], axis=1)
                        np_2_nii(gt_full, nii_gt_path, rotate90_hw=True)

                # Render comparison video: source | ground-truth | generated
                source_frames = tensor_to_video_frames(torch.from_numpy(source_video), min_val=-1, max_val=1)
                gt_frames = tensor_to_video_frames(torch.from_numpy(gt_video), min_val=-1, max_val=1)
                gen_frames = tensor_to_video_frames(torch.from_numpy(gen_np), min_val=-1, max_val=1)

                save_video_comparison(
                    source_frames,
                    gt_frames,
                    gen_frames,
                    os.path.join(video_dir, f"{sample_name}.mp4"),
                    fps=5
                )

                sample_idx += 1

    print(f"Generation completed! Results saved to: {output_dir}")
    return results


def tensor_to_video_frames(tensor, normalize=True,min_val=None,max_val=None):
    """
    Convert a tensor into a list of video frames.

    Args:
        tensor: tensor of shape (1, D, H, W)
        normalize: whether to normalize to [0, 255]

    Returns:
        List[np.ndarray]: list of video frames, each a (H, W) uint8 array
    """
    # Remove the batch dim: (1, D, H, W) -> (D, H, W)
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    frames = []
    for i in range(tensor.shape[0]):  # Iterate over the depth dimension
        frame = tensor[i].cpu().numpy()  # (H, W)

        if normalize:
            # If data is in [-1, 1], convert to [0, 1] first

            if min_val is None and max_val is None:
                frame = (frame-frame.min()) / (frame.max() - frame.min())
            else:
                frame = (frame+1) / 2

            # Convert to [0, 255]
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        
        frames.append(frame)
    
    return frames

def save_video_comparison(source_frames,original_frames, reconstructed_frames, output_path, fps=10):
    """
    Save a comparison of the original and reconstructed videos.

    Args:
        original_frames: list of original video frames
        reconstructed_frames: list of reconstructed video frames
        output_path: output path
        fps: video frame rate
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make sure both videos have the same number of frames
    min_frames = min(len(original_frames), len(reconstructed_frames), len(source_frames))
    original_frames = original_frames[:min_frames]
    reconstructed_frames = reconstructed_frames[:min_frames]
    source_frames = source_frames[:min_frames]

    # Build the comparison video (side by side)
    comparison_frames = []
    for source, orig, recon in zip(source_frames, original_frames, reconstructed_frames):
        # Make sure sizes match
        if orig.shape != recon.shape:
            recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))

        # Concatenate horizontally
        comparison = np.hstack([source,orig, recon])
        comparison_frames.append(comparison)

    # Save the comparison video
    comparison_path = output_path.parent / f"{output_path.stem}_comparison.mp4"
    imageio.mimsave(str(comparison_path), comparison_frames, fps=fps)
    print(f"Comparison video saved to: {comparison_path}")

    # Save the original and reconstructed videos separately
    # original_path = output_path.parent / f"{output_path.stem}_original.mp4"
    # reconstructed_path = output_path.parent / f"{output_path.stem}_reconstructed.mp4"

    original_path = output_path.parent / f"{output_path.stem}_gen.mp4"
    reconstructed_path = output_path.parent / f"{output_path.stem}_rec.mp4"
    
    
    imageio.mimsave(str(original_path), original_frames, fps=fps)
    imageio.mimsave(str(reconstructed_path), reconstructed_frames, fps=fps)
    
    print(f"Original video saved to: {original_path}")
    print(f"Reconstructed video saved to: {reconstructed_path}")
    
    return comparison_path, original_path, reconstructed_path



def main():
    parser = argparse.ArgumentParser(description='GPT Model Sampling Test')
    parser.add_argument('--config', type=str, 
                       default='configs/ldt_fsq_gen.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/generator.ckpt',
                       help='Path to GPT model checkpoint')
    parser.add_argument('--output_dir', type=str, default='output/ldt_gen',
                       help='Output directory for generated samples')
    parser.add_argument('--temperature', type=float, default=1.2,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Nucleus sampling parameter')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--save_nii', action='store_true', default=False,
                       help='If set, save generated/gt/source volumes as nii.gz files; otherwise skip it')
    
    args = parser.parse_args()

    setup_logging()
    
    print("=== GPT Model Sampling Test ===")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Device: {args.device}")
    
    # Check that the files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    try:
        # Load the config
        config = load_config(args.config)

        # Load the model
        model = load_model(args.checkpoint, config)

        # Build the dataset from the test section of the config
        dataset_config = config['data']['init_args']['test']['params']['config']

        dataset = CTVideoDataset(dataset_config)

        print(f"Dataset size: {len(dataset)}")

        # Run the sampling test
        print("\nStarting GPT sampling...")
        results = generate_samples_from_gpt(
            model=model,
            dataset=dataset,
            output_dir=args.output_dir,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            save_nii=args.save_nii
        )

        print("\n=== GPT Sampling completed successfully! ===")
        
    except Exception as e:
        print(f"Error during sampling: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Usage example:
# export CUDA_VISIBLE_DEVICES=0 python infer.py --config configs/ldt_gen.yaml --checkpoint checkpoints/ldt_gen/last.ckpt --max_samples 5