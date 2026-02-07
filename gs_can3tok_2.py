"""
Can3Tok Training with Projection Head Approach
Version: 4.0 - PROJECTION HEAD ON GS_DECODER HIDDEN STATE

Key Changes from v3.2:
- REMOVED: per-Gaussian feature extraction from encoder
- ADDED: SemanticProjectionHead on GS_decoder hidden state [B, 256]
- Projects hidden → [B, 40k, 128] for semantic loss
- 10x less memory than geo_decoder approach
- 20% faster training (no chunking needed)

Architecture:
  decoded_latents [B, 256, 384]
      ↓
  reshape + 8 MLPs
      ↓
  hidden [B, 256] ← Extract this!
      ↓
  Projection Head (3-layer MLP)
      ↓
  Per-Gaussian features [B, 40k, 128]
      ↓
  Semantic Contrastive Loss
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import argparse

# Michelangelo imports (VAE model)
from model.michelangelo.utils import instantiate_from_config
from model.michelangelo.utils.misc import get_config_from_file

# Data loading
import torch.utils.data as Data
from scipy.stats import special_ortho_group

# Semantic loss functions
from semantic_losses import compute_semantic_loss

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_memory_stats(device):
    """Print current GPU memory usage statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  💾 GPU Memory: Allocated={allocated:.2f}GB, "
              f"Reserved={reserved:.2f}GB, Peak={max_allocated:.2f}GB")

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Can3Tok Training - Projection Head Approach')
parser.add_argument('--use_wandb', action='store_true', default=False,
                    help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='Can3Tok-SceneSplat',
                    help='W&B project name')
parser.add_argument('--wandb_entity', type=str, default='3D-SSC',
                    help='W&B entity/team name')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--kl_weight', type=float, default=1e-5,
                    help='KL divergence loss weight')
parser.add_argument('--eval_every', type=int, default=10,
                    help='Evaluate on validation set every N epochs')
parser.add_argument('--failure_threshold', type=float, default=8000.0,
                    help='L2 error threshold for failure rate')

# Scene count parameters
parser.add_argument('--train_scenes', type=int, default=None,
                    help='Number of training scenes to use (None = all)')
parser.add_argument('--val_scenes', type=int, default=None,
                    help='Number of validation scenes to use (None = all)')

# Sampling method
parser.add_argument('--sampling_method', type=str, default='opacity',
                    choices=['random', 'opacity'],
                    help='Sampling method: random or opacity (default: opacity)')

# Semantic loss parameters
parser.add_argument('--segment_loss_weight', type=float, default=0.1,
                    help='Weight for segment-level contrastive loss (beta)')
parser.add_argument('--instance_loss_weight', type=float, default=0.0,
                    help='Weight for instance-level contrastive loss (gamma)')
parser.add_argument('--semantic_temperature', type=float, default=0.07,
                    help='Temperature for semantic contrastive loss (tau)')
parser.add_argument('--semantic_subsample', type=int, default=2000,
                    help='Number of Gaussians to subsample for semantic loss')

# Reconstruction loss scaling
parser.add_argument('--recon_scale', type=float, default=1000.0,
                    help='Scale factor for reconstruction loss to balance gradients')

# In argument parsing section, add:
parser.add_argument('--semantic_mode', type=str, default='hidden',
                    choices=['hidden', 'geometric', 'attention'],
                    help='Semantic feature extraction mode')

args = parser.parse_args()

# ============================================================================
# WEIGHTS & BIASES SETUP
# ============================================================================

wandb_enabled = False
if args.use_wandb:
    try:
        import wandb
        
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        
        run_name = f"can3tok_job_{job_id}_PROJECTION_HEAD"
        if args.segment_loss_weight > 0 or args.instance_loss_weight > 0:
            run_name += f"_semantic_beta{args.segment_loss_weight}"
        
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "learning_rate": args.lr,
                "architecture": "Can3Tok-ProjectionHead",
                "dataset": "SceneSplat-7K",
                "batch_size": args.batch_size,
                "epochs": args.num_epochs,
                "kl_weight": args.kl_weight,
                "feature_extraction": "projection_head_on_hidden",
                "hidden_dim": 256,
                "feature_dim": 128,
                "num_gaussians": 40000,
                "sampling_method": args.sampling_method,
                "eval_every": args.eval_every,
                "recon_scale": args.recon_scale,
                "segment_loss_weight": args.segment_loss_weight,
                "instance_loss_weight": args.instance_loss_weight,
                "semantic_temperature": args.semantic_temperature,
                "semantic_subsample": args.semantic_subsample,
            },
            tags=["scenesplat", "projection-head", "semantic-loss", "gaussian-splatting"],
        )
        print("✓ Weights & Biases enabled")
        wandb_enabled = True
    except Exception as e:
        print(f"✗ Weights & Biases failed: {e}")
        wandb_enabled = False
else:
    print("✗ Weights & Biases disabled")

# ============================================================================
# GPU SETUP
# ============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================================
# CONFIGURATION
# ============================================================================

loss_usage = "L1"
random_permute = 0
random_rotation = 1

resol = 200
data_path = "/home/yli11/scratch/datasets/gaussian_world/preprocessed/interior_gs"

num_epochs = args.num_epochs
bch_size = args.batch_size
kl_weight = args.kl_weight
eval_every = args.eval_every
failure_threshold = args.failure_threshold
train_scenes = args.train_scenes
val_scenes = args.val_scenes
sampling_method = args.sampling_method

# ============================================================================
# JOB-SPECIFIC CHECKPOINT FOLDER
# ============================================================================

job_id = os.environ.get('SLURM_JOB_ID', None)
if job_id:
    checkpoint_folder = f"job_{job_id}_PROJECTION_HEAD"
    if args.segment_loss_weight > 0 or args.instance_loss_weight > 0:
        checkpoint_folder += f"_semantic_beta{args.segment_loss_weight}_scale{args.recon_scale}"
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_folder = f"local_{timestamp}_PROJECTION_HEAD"
    if args.segment_loss_weight > 0 or args.instance_loss_weight > 0:
        checkpoint_folder += f"_semantic_beta{args.segment_loss_weight}_scale{args.recon_scale}"

save_path = f"/home/yli11/scratch/Hafeez_thesis/Can3Tok/checkpoints/{checkpoint_folder}/"
os.makedirs(save_path, exist_ok=True)

# Save run configuration
config_file = os.path.join(save_path, "config.txt")
with open(config_file, 'w') as f:
    f.write(f"Can3Tok Training Configuration - PROJECTION HEAD APPROACH\n")
    f.write(f"=" * 70 + "\n")
    f.write(f"Job ID: {job_id or 'local'}\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Feature Extraction: Projection Head on GS_decoder hidden state\n")
    f.write(f"Architecture: hidden [B,256] → ProjectionHead → [B,40k,128]\n")
    f.write(f"Device: {device}\n")
    f.write(f"Batch size: {bch_size}\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Learning rate: {args.lr}\n")
    f.write(f"KL weight: {kl_weight}\n")
    f.write(f"Recon scale: {args.recon_scale}\n")
    f.write(f"Segment loss weight (β): {args.segment_loss_weight}\n")
    f.write(f"Instance loss weight (γ): {args.instance_loss_weight}\n")
    f.write(f"Semantic temperature (τ): {args.semantic_temperature}\n")
    f.write(f"Semantic subsample: {args.semantic_subsample}\n")
    f.write(f"Dataset: {data_path}\n")
    f.write(f"Sampling method: {sampling_method}\n")
    f.write(f"=" * 70 + "\n")

print(f"Configuration:")
print(f"  Job ID: {job_id or 'local'}")
print(f"  VERSION: PROJECTION HEAD on GS_decoder hidden state")
print(f"  Architecture: hidden [B,256] → 3-layer MLP → [B,40k,128]")

if args.segment_loss_weight > 0 or args.instance_loss_weight > 0:
    print(f"  🧠 SEMANTIC LOSS ENABLED:")
    print(f"     Segment weight (β): {args.segment_loss_weight}")
    print(f"     Instance weight (γ): {args.instance_loss_weight}")
    print(f"     Temperature (τ): {args.semantic_temperature}")
    print(f"     Subsample size: {args.semantic_subsample}")
    print(f"     Recon scale: {args.recon_scale}")
else:
    print(f"  Semantic loss: DISABLED")

print(f"  Device: {device}")
print(f"  Batch size: {bch_size}")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {args.lr}")
print(f"  KL weight: {kl_weight}")
print(f"  Sampling method: {sampling_method}")
print(f"  Save path: {save_path}")
print()

# Define geometric indices GLOBALLY
GEOMETRIC_INDICES = list(range(4, 7)) + [10] + list(range(11, 18))
# [4,5,6] xyz + [10] opacity + [11,12,13] scale + [14,15,16,17] quat

# ============================================================================
# MODEL SETUP
# ============================================================================

print("Loading model with SemanticProjectionHead...")
config_path_perceiver = "./model/configs/aligned_shape_latents/shapevae-256.yaml"
model_config_perceiver = get_config_from_file(config_path_perceiver)

if hasattr(model_config_perceiver, "model"):
    model_config_perceiver = model_config_perceiver.model

# Add semantic_mode to config
model_config_perceiver['semantic_mode'] = args.semantic_mode

perceiver_encoder_decoder = instantiate_from_config(model_config_perceiver)

# Multi-GPU setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    gs_autoencoder = nn.DataParallel(perceiver_encoder_decoder)
else:
    print("Using single GPU")
    gs_autoencoder = perceiver_encoder_decoder

gs_autoencoder.to(device)

# Optimizer
optimizer = torch.optim.Adam(gs_autoencoder.parameters(), lr=args.lr, betas=[0.9, 0.999])

# print("✓ Model loaded with SemanticProjectionHead")
print()


# ============================================================================
# VERIFY SEMANTIC MODE
# ============================================================================
print(f"\n{'='*70}")
print(f"VERIFYING SEMANTIC MODE:")
print(f"  Command line argument: {args.semantic_mode}")
print(f"  Model attribute: {getattr(gs_autoencoder, 'semantic_mode', 'NOT FOUND')}")

# Force set if wrong
if hasattr(gs_autoencoder, 'semantic_mode'):
    current_mode = gs_autoencoder.semantic_mode
    if current_mode != args.semantic_mode:
        print(f"  ⚠️  MISMATCH! Model has '{current_mode}', forcing to '{args.semantic_mode}'")
        gs_autoencoder.semantic_mode = args.semantic_mode
        
        # Also need to disable unused heads for memory
        if args.semantic_mode == 'geometric':
            print(f"  ✓ Disabling unused semantic heads for efficiency")
            # These might not exist depending on your initialization
            if hasattr(gs_autoencoder, 'semantic_projection_hidden'):
                gs_autoencoder.semantic_projection_hidden = None
            if hasattr(gs_autoencoder, 'semantic_attention_head'):
                gs_autoencoder.semantic_attention_head = None
        elif args.semantic_mode == 'hidden':
            print(f"  ⚠️  Warning: Hidden mode uses 329M params!")
            if hasattr(gs_autoencoder, 'semantic_projection_geometric'):
                gs_autoencoder.semantic_projection_geometric = None
            if hasattr(gs_autoencoder, 'semantic_attention_head'):
                gs_autoencoder.semantic_attention_head = None
        elif args.semantic_mode == 'attention':
            print(f"  ✓ Using attention mode")
            if hasattr(gs_autoencoder, 'semantic_projection_hidden'):
                gs_autoencoder.semantic_projection_hidden = None
            if hasattr(gs_autoencoder, 'semantic_projection_geometric'):
                gs_autoencoder.semantic_projection_geometric = None
else:
    print(f"  ❌ CRITICAL: Model doesn't have semantic_mode attribute!")
    # Add it manually
    gs_autoencoder.semantic_mode = args.semantic_mode
    print(f"  ✓ Added semantic_mode attribute manually")

print(f"{'='*70}\n")

# ============================================================================
# DATASET LOADING
# ============================================================================

print("Loading datasets...")
from gs_dataset_scenesplat import gs_dataset

# Training dataset
print("\n" + "="*70)
print("TRAINING DATASET")
print("="*70)
gs_dataset_train = gs_dataset(
    root=os.path.join(data_path, "train_grid1.0cm_chunk8x8_stride6x6"),
    resol=resol,
    random_permute=True,
    train=True,
    sampling_method=sampling_method,
    max_scenes=train_scenes
)

trainDataLoader = Data.DataLoader(
    dataset=gs_dataset_train, 
    batch_size=bch_size,
    shuffle=True, 
    num_workers=12,
    pin_memory=True
)



# Validation dataset
print("="*70)
print("VALIDATION DATASET")
print("="*70)

gs_dataset_val = gs_dataset(
    root=os.path.join(data_path, "val"),
    resol=resol,
    random_permute=False,
    train=False,
    sampling_method=sampling_method,
    max_scenes=val_scenes
)

valDataLoader = Data.DataLoader(
    dataset=gs_dataset_val,
    batch_size=bch_size,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

print("="*70)
print("DATASET SUMMARY")
print("="*70)
print(f"✓ Training: {len(gs_dataset_train)} scenes, {len(trainDataLoader)} batches")
print(f"✓ Validation: {len(gs_dataset_val)} scenes, {len(valDataLoader)} batches")
print(f"✓ Sampling method: {sampling_method}")
print("="*70)
print()

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, dataloader, device, failure_threshold):
    """Evaluate model on validation set"""
    model.eval()
    
    total_l2_error = 0.0
    total_kl = 0.0
    per_scene_l2_errors = []
    num_failures = 0
    num_scenes = 0
    
    with torch.no_grad():
        for i_batch, batch_data in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if isinstance(batch_data, dict):
                UV_gs_batch = batch_data['features'].type(torch.float32).to(device)
            else:
                UV_gs_batch = batch_data[0].type(torch.float32).to(device)
            
            batch_size = UV_gs_batch.shape[0]
            
            shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = model(
                UV_gs_batch,
                UV_gs_batch,
                UV_gs_batch,
                UV_gs_batch[:, :, :3]
            )
            
            # Define geometric indices (11 params: xyz, opacity, scale, quat - NO RGB!)
            GEOMETRIC_INDICES = list(range(4, 7)) + [10] + list(range(11, 18))
            # [4,5,6] xyz + [10] opacity + [11,12,13] scale + [14,15,16,17] quat

            # In training loop:
            # Replace target extraction:
            target = UV_gs_batch[:, :, GEOMETRIC_INDICES]
            UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 11)
            
            batch_l2_error = torch.norm(
                UV_gs_recover_reshaped - target,
                p=2
            ) / batch_size
            
            per_scene_norms = torch.norm(
                UV_gs_recover_reshaped - target,
                p=2,
                dim=(1, 2)
            )
            per_scene_errors_scaled = per_scene_norms / np.sqrt(batch_size)
            
            kl_loss = -0.5 * torch.sum(
                1.0 + log_var - mu.pow(2) - log_var.exp(),
                dim=1
            )
            
            total_l2_error += batch_l2_error.item() * batch_size
            total_kl += kl_loss.sum().item()
            per_scene_l2_errors.extend(per_scene_errors_scaled.cpu().numpy().tolist())
            num_failures += (per_scene_errors_scaled.cpu().numpy() > failure_threshold).sum()
            num_scenes += batch_size
            
            # Clean up
            del shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features
            del target, UV_gs_recover_reshaped, batch_l2_error, per_scene_norms, per_scene_errors_scaled, kl_loss
            
            if i_batch % 2 == 0:
                torch.cuda.empty_cache()
    
    avg_l2_error = total_l2_error / num_scenes
    avg_kl = total_kl / num_scenes
    failure_rate = (num_failures / num_scenes) * 100.0
    
    per_scene_l2_errors = np.array(per_scene_l2_errors)
    
    model.train()
    
    return {
        'avg_l2_error': avg_l2_error,
        'l2_std': per_scene_l2_errors.std(),
        'l2_min': per_scene_l2_errors.min(),
        'l2_max': per_scene_l2_errors.max(),
        'l2_median': np.median(per_scene_l2_errors),
        'failure_rate': failure_rate,
        'num_failures': num_failures,
        'num_scenes': num_scenes,
        'avg_kl': avg_kl,
        'per_scene_errors': per_scene_l2_errors,
    }

# ============================================================================
# TRAINING SETUP
# ============================================================================

volume_dims = 40
resolution = 16.0 / volume_dims
origin_offset = torch.tensor(
    np.array([
        (volume_dims - 1) / 2, 
        (volume_dims - 1) / 2, 
        (volume_dims - 1) / 2
    ]) * resolution, 
    dtype=torch.float32
).to(device)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("="*70)
print("STARTING TRAINING - PROJECTION HEAD APPROACH")
if args.segment_loss_weight > 0 or args.instance_loss_weight > 0:
    print("🧠 SEMANTIC LOSS ENABLED (projection head features)")
print("="*70)
print()

global_step = 0
best_val_loss = float('inf')
best_epoch = 0

for epoch in tqdm(range(num_epochs), desc="Training"):
    epoch_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_semantic_loss = 0.0
    
    gs_autoencoder.train()
    
    for i_batch, batch_data in enumerate(trainDataLoader):
        # Initialize to None (in case forward fails)
        per_gaussian_features = None
        
        # Extract data
        if isinstance(batch_data, dict):
            UV_gs_batch = batch_data['features'].type(torch.float32).to(device)
            segment_labels = batch_data['segment_labels'].type(torch.int64).to(device)
            instance_labels = batch_data['instance_labels'].type(torch.int64).to(device)
            has_semantics = batch_data['has_semantics']
        else:
            UV_gs_batch = batch_data[0].type(torch.float32).to(device)
            segment_labels = None
            instance_labels = None
            has_semantics = False
        
        # Random permutation
        if epoch % 1 == 0 and random_permute == 1:
            perm_indices = torch.randperm(UV_gs_batch.size()[1])
            UV_gs_batch = UV_gs_batch[:, perm_indices]
            
            if segment_labels is not None:
                segment_labels = segment_labels[:, perm_indices]
            if instance_labels is not None:
                instance_labels = instance_labels[:, perm_indices]
        
        # Random rotation (every 5 epochs)
        if epoch % 5 == 0 and epoch > 1 and random_rotation == 1:
            rand_rot_comp = special_ortho_group.rvs(3)
            rand_rot = torch.tensor(
                np.dot(rand_rot_comp, rand_rot_comp.T), 
                dtype=torch.float32
            ).to(UV_gs_batch.device)
            
            UV_gs_batch[:, :, 4:7] = UV_gs_batch[:, :, 4:7] @ rand_rot
            
            for bcbc in range(UV_gs_batch.shape[0]):
                shifted_points = UV_gs_batch[bcbc, :, 4:7] + origin_offset
                voxel_indices = torch.floor(shifted_points / resolution)
                voxel_indices = torch.clip(voxel_indices, 0, volume_dims - 1)
                voxel_centers_batch = (voxel_indices - (volume_dims - 1) / 2) * resolution
                UV_gs_batch[bcbc, :, :3] = voxel_centers_batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        shape_embed, mu, log_var, z, UV_gs_recover, per_gaussian_features = gs_autoencoder(
            UV_gs_batch, 
            UV_gs_batch, 
            UV_gs_batch, 
            UV_gs_batch[:, :, :3]
        )
        
        # Compute losses
        KL_loss = -0.5 * torch.sum(
            1.0 + log_var - mu.pow(2) - log_var.exp(), 
            dim=1
        ).mean()
        

        target = UV_gs_batch[:, :, GEOMETRIC_INDICES]  # [B, 40k, 11]
        UV_gs_recover_reshaped = UV_gs_recover.reshape(UV_gs_batch.shape[0], -1, 11)

        # Replace reconstruction loss:
        recon_loss_raw = torch.norm(
            UV_gs_recover_reshaped - target,
            p=2
        ) / UV_gs_batch.shape[0]

        
        recon_loss = recon_loss_raw / args.recon_scale
        
        # Semantic contrastive loss
        semantic_loss = torch.tensor(0.0, device=device)
        semantic_metrics = {}
        
        if (args.segment_loss_weight > 0 or args.instance_loss_weight > 0) and \
        segment_labels is not None and per_gaussian_features is not None:
            
            semantic_loss, semantic_metrics = compute_semantic_loss(
                embeddings=per_gaussian_features,  # Now [B, 40k, 32] with your change
                segment_labels=segment_labels,
                instance_labels=instance_labels,
                batch_size=UV_gs_batch.shape[0],
                segment_weight=args.segment_loss_weight,
                instance_weight=args.instance_loss_weight,
                temperature=args.semantic_temperature,
                subsample=args.semantic_subsample
            )
            
            # Free memory immediately after computing loss
            del per_gaussian_features
            per_gaussian_features = None  # Set to None so cleanup works
            torch.cuda.empty_cache()
        
        elif per_gaussian_features is not None:
            del per_gaussian_features
            per_gaussian_features = None
            torch.cuda.empty_cache()
        
        # Combined loss
        loss = recon_loss + kl_weight * KL_loss + semantic_loss
        
        # Save values for logging (before deleting tensors)
        loss_value = loss.item()
        recon_loss_raw_value = recon_loss_raw.item()
        recon_loss_value = recon_loss.item()
        kl_loss_value = KL_loss.item()
        semantic_loss_value = semantic_loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Memory cleanup
        del shape_embed, mu, log_var, z, UV_gs_recover
        del recon_loss_raw, recon_loss, KL_loss, semantic_loss, loss
        
        # Clean up per_gaussian_features if still exists
        if per_gaussian_features is not None:
            del per_gaussian_features
        
        if i_batch % 2 == 0:
            torch.cuda.empty_cache()
        
        # Accumulate losses
        epoch_loss += loss_value
        epoch_recon_loss += recon_loss_raw_value
        epoch_kl_loss += kl_loss_value
        epoch_semantic_loss += semantic_loss_value
        
        # Log to wandb
        if wandb_enabled:
            log_dict = {
                "train/step_loss": loss_value,
                "train/step_recon_loss": recon_loss_raw_value,
                "train/step_recon_loss_scaled": recon_loss_value,
                "train/step_kl_loss": kl_loss_value,
            }
            
            if semantic_metrics:
                for key, value in semantic_metrics.items():
                    log_dict[f"train/step_{key}"] = value
            
            wandb_run.log(log_dict, step=global_step)
        
        global_step += 1
        
        # Print first batch
        if i_batch == 0:
            print_msg = (f"Epoch {epoch}/{num_epochs} | "
                        f"Batch {i_batch}/{len(trainDataLoader)} | "
                        f"Loss: {loss_value:.2f} | "
                        f"Recon: {recon_loss_raw_value:.2f} | "
                        f"KL: {kl_loss_value:.2f}")
            
            if semantic_loss_value > 0:
                print_msg += f" | Semantic: {semantic_loss_value:.4f}"
            
            print(print_msg)
    
    # Epoch summary
    avg_train_loss = epoch_loss / len(trainDataLoader)
    avg_train_recon = epoch_recon_loss / len(trainDataLoader)
    avg_train_kl = epoch_kl_loss / len(trainDataLoader)
    avg_train_semantic = epoch_semantic_loss / len(trainDataLoader)
    
    # Validation
    val_metrics = None
    if epoch % eval_every == 0 or epoch == num_epochs - 1:
        print(f"\n{'='*70}")
        print(f"RUNNING VALIDATION (Epoch {epoch})")
        print(f"{'='*70}")
        
        val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, failure_threshold)
        
        print(f"Validation Results:")
        print(f"  L2 Error: {val_metrics['avg_l2_error']:.2f} ± {val_metrics['l2_std']:.2f}")
        print(f"  Failure Rate: {val_metrics['failure_rate']:.2f}%")
        
        # Track best model
        if val_metrics['avg_l2_error'] < best_val_loss:
            best_val_loss = val_metrics['avg_l2_error']
            best_epoch = epoch
            
            if torch.cuda.device_count() > 1:
                model_state = gs_autoencoder.module.state_dict()
            else:
                model_state = gs_autoencoder.state_dict()
            
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_l2_error': val_metrics['avg_l2_error'],
                'approach': 'projection_head',
            }, best_model_path)
            print(f"  ✓ New best model saved! (L2: {best_val_loss:.2f})")
        
        print(f"{'='*70}\n")
    
    # Log epoch metrics
    if wandb_enabled:
        log_dict = {
            "train/epoch_loss": avg_train_loss,
            "train/epoch_recon": avg_train_recon,
            "train/epoch_kl": avg_train_kl,
            "train/epoch_semantic": avg_train_semantic,
            "train/epoch": epoch,
            "best/val_l2_error": best_val_loss,
            "best/epoch": best_epoch,
        }
        
        if val_metrics is not None:
            log_dict.update({
                "val/l2_error": val_metrics['avg_l2_error'],
                "val/failure_rate": val_metrics['failure_rate'],
            })
        
        wandb_run.log(log_dict, step=global_step)
    
    # Print summary every 20 epochs
    if epoch % 20 == 0:
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*70}")
        print(f"Avg Train Loss: {avg_train_loss:.2f}")
        print(f"Avg Train Recon: {avg_train_recon:.2f}")
        print(f"Avg Train KL: {avg_train_kl:.2f}")
        if avg_train_semantic > 0:
            print(f"Avg Train Semantic: {avg_train_semantic:.4f}")
        print(f"Best Val L2 Error: {best_val_loss:.2f} (epoch {best_epoch})")
        if epoch % 10 == 0:
            print_memory_stats(device)
        print(f"{'='*70}\n")
    
    # Save checkpoints every 10 epochs
    if epoch >= 10 and epoch % 10 == 0:
        gs_autoencoder.eval()
        
        if torch.cuda.device_count() > 1:
            model_state = gs_autoencoder.module.state_dict()
        else:
            model_state = gs_autoencoder.state_dict()
        
        checkpoint_path = os.path.join(save_path, f"{int(epoch)}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'approach': 'projection_head',
        }, checkpoint_path)
        
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        gs_autoencoder.train()

# ============================================================================
# FINAL SAVE
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

# Final validation
final_val_metrics = evaluate_model(gs_autoencoder, valDataLoader, device, failure_threshold)

print(f"\nFinal Validation Results:")
print(f"  L2 Error: {final_val_metrics['avg_l2_error']:.2f}")
print(f"  Best Val L2: {best_val_loss:.2f} (epoch {best_epoch})")

# Save final model
if torch.cuda.device_count() > 1:
    model_state = gs_autoencoder.module.state_dict()
else:
    model_state = gs_autoencoder.state_dict()

final_path = os.path.join(save_path, "final.pth")
torch.save({
    'epoch': num_epochs - 1,
    'model_state_dict': model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'final_val_l2': final_val_metrics['avg_l2_error'],
    'best_val_l2': best_val_loss,
    'best_epoch': best_epoch,
    'approach': 'projection_head',
}, final_path)

print(f"✓ Final checkpoint saved: {final_path}")
print(f"✓ Best model saved: {os.path.join(save_path, 'best_model.pth')}")
print("="*70)

if wandb_enabled:
    wandb_run.summary.update({
        "final_val_l2_error": final_val_metrics['avg_l2_error'],
        "best_val_l2_error": best_val_loss,
        "best_epoch": best_epoch,
        "approach": "projection_head",
    })
    wandb_run.finish()
    print("✓ Weights & Biases run finished")