
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler  # AMP 混合精度
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import WheatDataset
from model import SpectralMambaClassifier, S3M_MAE
from tqdm import tqdm
import math

# 启用 FlashAttention (PyTorch 2.0+ SDPA backend)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

def setup_ddp():
    """Initialize DDP if environment variables are set (by torchrun)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def save_checkpoint_atomic(checkpoint_dict, save_path):
    """Atomically save checkpoint to reduce corruption risk on interrupted runs."""
    tmp_path = f"{save_path}.tmp"
    torch.save(checkpoint_dict, tmp_path)
    os.replace(tmp_path, save_path)


def resolve_resume_checkpoint(args):
    """Resolve resume checkpoint path with primary and backup fallback file."""
    # Highest priority: explicit --resume if valid.
    if args.resume:
        if os.path.exists(args.resume):
            return args.resume
        print(f"[WARNING] Provided --resume path not found: {args.resume}")

    # Primary auto-resume path.
    primary_last = os.path.join(args.output_dir, "checkpoint_last.pth")
    if os.path.exists(primary_last):
        return primary_last

    # Backup fallback checkpoint path.
    backup_path = args.backup_resume_path
    if backup_path and os.path.exists(backup_path):
        return backup_path

    return None

# Cosine Learning Rate Scheduler with Warmup (Step-wise)
def adjust_learning_rate(optimizer, epoch, i, len_loader, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # Calculate fractional epoch
    cur_step = epoch * len_loader + i
    warmup_steps = args.warmup_epochs * len_loader
    total_steps = args.epochs * len_loader
    
    if cur_step < warmup_steps:
        # Linear Warmup from 1e-6 to args.lr
        start_lr = 1e-6
        lr = start_lr + (args.lr - start_lr) * cur_step / warmup_steps
    else:
        # Cosine Decay
        progress = (cur_step - warmup_steps) / (total_steps - warmup_steps)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def pretrain(args):
    # DDP Setup
    rank, world_size, local_rank, use_ddp = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"Using device: {device} for Pre-training")
        print(f"DDP: {'Enabled' if use_ddp else 'Disabled'} (World Size: {world_size})")
        print("[INFO] FlashAttention (SDPA) backends enabled.")

    # 1. Dataset
    train_dataset = WheatDataset(root_dir=args.train_dir, transform=None, mode='all', return_name=False, expand_data=False)
    test_dataset = WheatDataset(root_dir=args.test_dir, transform=None, mode='all', return_name=False, expand_data=False)
    
    # Combined List
    datasets_to_concat = [train_dataset, test_dataset]
    
    # Optional: Extra Data (Gen Patches)
    if args.extra_data_dir and os.path.exists(args.extra_data_dir):
        print(f"[INFO] Loading Extra Data from: {args.extra_data_dir}")
        extra_dataset = WheatDataset(root_dir=args.extra_data_dir, transform=None, mode='all', return_name=False, expand_data=False)
        datasets_to_concat.append(extra_dataset)
        print(f"[INFO] Added {len(extra_dataset)} extra samples.")
    
    # Combine
    full_dataset = torch.utils.data.ConcatDataset(datasets_to_concat)
    data_fingerprint = {
        'train_dir': os.path.abspath(args.train_dir),
        'test_dir': os.path.abspath(args.test_dir),
        'extra_data_dir': os.path.abspath(args.extra_data_dir) if args.extra_data_dir else "",
        'full_dataset_size': int(len(full_dataset))
    }
    if is_main_process():
        print(f"Pre-training Dataset Size: {len(full_dataset)}")
    
    # DDP Sampler
    sampler = DistributedSampler(full_dataset, shuffle=True) if use_ddp else None
    dataloader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None),  # shuffle=False when using sampler
        sampler=sampler,
        num_workers=args.num_workers, 
        drop_last=True,
        pin_memory=True
    )
    
    # 2. Model
    classifier = SpectralMambaClassifier(
        num_classes=3, 
        d_model=args.d_model, 
        depth=args.depth, 
        dropout=0.0, 
        drop_path_rate=0.0,
        use_checkpoint=args.gradient_checkpoint  # Memory optimization
    )
    model = S3M_MAE(encoder=classifier, decoder_dim=args.decoder_dim, decoder_depth=args.decoder_depth, spectral_grad_weight=args.spectral_grad_weight)
    model = model.to(device)
    
    

    # AMP 设置
    use_amp = args.amp and device.type == 'cuda'
    # CRITICAL: T4 (SM 7.5) does NOT have native BF16 Tensor Cores!
    # torch.cuda.is_bf16_supported() returns True but it's SOFTWARE emulation (extremely slow).
    # Force FP16 which T4 natively supports.
    amp_dtype = torch.float16
    scaler = GradScaler(enabled=use_amp)
    
    if use_amp:
        if is_main_process():
            print(f"[INFO] AMP Enabled with dtype={amp_dtype}")
    
    # DDP Wrap (find_unused_parameters=True for MAE, since classification head is not used)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process():
            print("[INFO] Model wrapped with DDP")
    
    # 3. Optimizer
    eff_batch_size = args.batch_size * args.grad_accum_steps * world_size
    args.lr = args.base_lr * eff_batch_size / 256
    if is_main_process():
        print(f"Effective Batch Size: {eff_batch_size}, Effective LR: {args.lr}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    # 4. Training Loop
    if is_main_process():
        print(f"Starting Pre-training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    last_epoch_loss = None
    patience_counter = 0
    start_epoch = 0
    
    # --- Resume Logic ---
    # Auto-resume from primary/backup locations.
    original_resume = args.resume
    args.resume = resolve_resume_checkpoint(args)
    if args.resume is not None and original_resume != args.resume:
        print(f"[INFO] Auto-resuming from FOUND last checkpoint: {args.resume}")

    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"[ERROR] Resume checkpoint is not a dict: {args.resume}. "
                f"This file cannot restore optimizer/epoch/best_loss."
            )
        
        ckpt_type = checkpoint.get('checkpoint_type', 'unknown')
        if ckpt_type not in ['pretrain_full', 'unknown']:
            print(f"[WARNING] checkpoint_type={ckpt_type}, expected pretrain_full.", flush=True)
        
        ckpt_fp = checkpoint.get('data_fingerprint')
        if isinstance(ckpt_fp, dict) and ckpt_fp != data_fingerprint:
            print("[WARNING] Resume checkpoint data fingerprint mismatch.", flush=True)
            print(f"  Current: {data_fingerprint}", flush=True)
            print(f"  Checkpoint: {ckpt_fp}", flush=True)
        
        # Load Model (Handle DDP module. prefix mismatch)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Auto-fix DDP prefix mismatch
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            
            if len(model_keys & ckpt_keys) == 0:
                # No overlap - likely prefix mismatch
                # Check if model expects "module." but checkpoint doesn't have it
                sample_model_key = next(iter(model_keys))
                sample_ckpt_key = next(iter(ckpt_keys))
                
                if sample_model_key.startswith("module.") and not sample_ckpt_key.startswith("module."):
                    # Add "module." prefix to checkpoint keys
                    state_dict = {f"module.{k}": v for k, v in state_dict.items()}
                    print("[INFO] Added 'module.' prefix to checkpoint keys for DDP compatibility.")
                elif not sample_model_key.startswith("module.") and sample_ckpt_key.startswith("module."):
                    # Strip "module." prefix from checkpoint keys
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                    print("[INFO] Stripped 'module.' prefix from checkpoint keys.")
            
            # Load with strict=False to allow architecture updates (e.g. spectral_adapter removal)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
            if len(msg.missing_keys) > 0:
                print(f"[INFO] Example missing: {msg.missing_keys[:5]}")
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print("[WARNING] optimizer_state_dict missing in checkpoint. Optimizer resumed from fresh init.")
            if 'scaler_state_dict' in checkpoint and scaler.is_enabled() and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            ckpt_epoch = checkpoint.get('epoch', -1)
            try:
                start_epoch = int(ckpt_epoch) + 1
            except Exception:
                print(f"[WARNING] Invalid epoch in checkpoint: {ckpt_epoch}. Fallback start_epoch=0.")
                start_epoch = 0

            loaded_best = checkpoint.get('best_loss', float('inf'))
            try:
                best_loss = float(loaded_best)
            except Exception:
                print(f"[WARNING] Invalid best_loss in checkpoint: {loaded_best}. Fallback to inf.")
                best_loss = float('inf')

            last_epoch_loss = checkpoint.get('last_epoch_loss', None)
            args.mask_ratio = checkpoint.get('mask_ratio', args.mask_ratio)
            last_loss_msg = f"{last_epoch_loss:.4f}" if last_epoch_loss is not None else "N/A"
            print(
                f"  > Resumed at Epoch {start_epoch}, "
                f"Best Epoch Loss: {best_loss:.4f}, "
                f"Last Completed Epoch Loss: {last_loss_msg}, "
                f"Mask Ratio: {args.mask_ratio}"
            )
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(
                "  > Loaded model weights only (no optimizer/epoch/best_loss in file). "
                "start_epoch=0, best_loss=inf."
            )
            
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler (important for shuffling)
        if use_ddp:
            sampler.set_epoch(epoch)
            
        model.train()
        running_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{args.epochs}", disable=not is_main_process())
        optimizer.zero_grad()
        
        for i, (imgs, _) in enumerate(loop):
            # Adjust LR per step
            lr = adjust_learning_rate(optimizer, epoch, i, len(dataloader), args)
            
            # Ignore labels
            imgs = imgs.to(device)
            
            # Forward with AMP
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                batch_loss, _, _ = model(
                    imgs,
                    mask_ratio=args.mask_ratio,
                    spectral_mask_ratio=args.spectral_mask_ratio
                )
                loss = batch_loss / args.grad_accum_steps
            
            # Backward
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            batch_loss_value = float(batch_loss.detach().item())
            running_loss += batch_loss_value
            loop.set_postfix(loss=batch_loss_value, lr=f"{lr:.2e}")
            
        avg_loss = running_loss / len(dataloader)
        last_epoch_loss = avg_loss
        
        # Sync loss across processes for accurate reporting
        if use_ddp:
            avg_loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        
        if is_main_process():
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
        # Early Stopping & Saving (only on main process)
        if is_main_process():
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                print(f"[INFO] New Best Loss! best_loss updated to {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"[INFO] No improvement. Patience: {patience_counter}/{args.patience}")

            # Save encoder "last" weights every epoch (for fine-tuning preload)
            save_path = os.path.join(args.output_dir, "mae_pretrain_last.pth")
            encoder_to_save = model.module.encoder if use_ddp else model.encoder
            torch.save(encoder_to_save.state_dict(), save_path)
            print(f"[INFO] Saved LAST encoder weights to {save_path}")
        
        # Sync patience_counter across all processes for consistent early stopping
        if use_ddp:
            patience_tensor = torch.tensor([patience_counter], device=device)
            dist.broadcast(patience_tensor, src=0)
            patience_counter = int(patience_tensor.item())
            
        if patience_counter >= args.patience:
            if is_main_process():
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
            break
            
        # Adaptive Masking Strategy
        if avg_loss < 0.02:
            current_mask = args.mask_ratio
            if current_mask < 0.90:
                args.mask_ratio = min(0.90, current_mask + 0.05)
                if is_main_process():
                    print(f"[ADAPTIVE] Loss {avg_loss:.4f} < 0.02. Increasing Mask Ratio: {current_mask:.2f} -> {args.mask_ratio:.2f}")
        
        # Save Checkpoint (only on main process)
        if is_main_process():
            checkpoint_dict = {
                'checkpoint_type': 'pretrain_full',
                'epoch': epoch,
                'model_state_dict': (model.module if use_ddp else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'best_loss': best_loss,
                'last_epoch_loss': last_epoch_loss,
                'mask_ratio': args.mask_ratio,
                'data_fingerprint': data_fingerprint
            }
            save_checkpoint_atomic(checkpoint_dict, os.path.join(args.output_dir, "checkpoint_last.pth"))

            if (epoch + 1) == args.epochs:
                full_save_path = os.path.join(args.output_dir, "checkpoint_final.pth")
                save_checkpoint_atomic(checkpoint_dict, full_save_path)
                print(f"Saved Final Checkpoint to {full_save_path}")
    
    # Cleanup DDP
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/HS")
    parser.add_argument("--test_dir", type=str, default="/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/val/HS")
    parser.add_argument("--extra_data_dir", type=str, default=None, help="Optional: Path to additional augmented data (e.g. generated patches)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/")
    
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--backup_resume_path",
        type=str,
        default="/kaggle/input/datasets/xishengfeng/checkpoint20/checkpoint_last.pth",
        help="Backup checkpoint_last.pth path used when auto-resume cannot find output_dir/checkpoint_last.pth",
    )

    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--save_freq", type=int, default=999) # Disabled
    
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    
    parser.add_argument("--d_model", type=int, default=512) # Must match finetune
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--spectral_mask_ratio", type=float, default=0.15, help="Ratio of contiguous spectral bands to mask (0=disabled)")
    parser.add_argument("--spectral_grad_weight", type=float, default=0.1, help="Weight for spectral gradient loss (gamma)")
    
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    
    # Acceleration Options
    parser.add_argument("--amp", action='store_true', default=True, help="Enable AMP mixed precision")
    parser.add_argument("--no_amp", action='store_false', dest='amp', help="Disable AMP")

    # Memory Optimization
    parser.add_argument("--gradient_checkpoint", action='store_true', default=False, help="Enable gradient checkpointing (saves ~30-40%% VRAM)")
    parser.add_argument("--no_gradient_checkpoint", action='store_false', dest='gradient_checkpoint', help="Disable gradient checkpointing")
    
    args = parser.parse_args()

    # In Kaggle, always write outputs directly to root for easier manual download.
    kaggle_root = "/kaggle/working"
    if os.path.isdir(kaggle_root):
        if os.path.abspath(args.output_dir) != os.path.abspath(kaggle_root):
            print(f"[INFO] Override output_dir to Kaggle root: {kaggle_root}")
        args.output_dir = kaggle_root
    
    # Create output dir (exist_ok for DDP compatibility)
    os.makedirs(args.output_dir, exist_ok=True)
        
    pretrain(args)
