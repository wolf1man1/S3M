
import os
import contextlib
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import WheatDataset
from model import SpectralMambaClassifier, inject_lora
from consts import CLASS_MAP
from tqdm import tqdm

import pandas as pd
import numpy as np

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
# Helpers removed as augmentation is now handled in Dataset with expand_data=True

def inference_loop(model, test_loader, device, epoch, output_dir, tta=False):
    model.eval()
    results = []
    print(f"Running Inference for Epoch {epoch+1} (TTA={tta})...")
    
    # Inverse Class Map: 0->Health, 1->Rust, 2->Other
    inv_map = {v: k for k, v in CLASS_MAP.items()}
    
    with torch.no_grad():
        for batch_data, _, filenames in tqdm(test_loader, desc="Inference"):
            batch_data = batch_data.to(device)
            
            if tta:
                # TTA: Original + HFlip + VFlip
                out1 = model(batch_data)
                out2 = model(torch.flip(batch_data, [3])) # HFlip
                out3 = model(torch.flip(batch_data, [2])) # VFlip
                outputs = (out1 + out2 + out3) / 3.0
            else:
                outputs = model(batch_data)
                
            _, predicted = torch.max(outputs.data, 1)
            
            for i in range(len(filenames)):
                fname = filenames[i]
                pred_idx = predicted[i].item()
                pred_label = inv_map.get(pred_idx, "Other")
                results.append({"Id": fname, "Category": pred_label})
    
    df = pd.DataFrame(results)
    # Sort by Id just in case
    df = df.sort_values("Id")
    
    save_path = os.path.join(output_dir, f"submission_epoch_{epoch+1}.csv")
    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

def evaluate(model, loader, criterion, device, tta=False):
    """
    Evaluates the model on a clean dataset (no mixup/smote).
    Returns: Average Loss, Accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data, batch_labels, _ in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            if tta:
                # TTA: Original + HFlip + VFlip
                out1 = model(batch_data)
                out2 = model(torch.flip(batch_data, [3])) # HFlip
                out3 = model(torch.flip(batch_data, [2])) # VFlip
                outputs = (out1 + out2 + out3) / 3.0
            else:
                outputs = model(batch_data)
                
            loss = criterion(outputs, batch_labels)
            
            # Sum up batch loss (multiply by batch size since CrossEntropy is mean by default)
            running_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
    avg_loss = running_loss / total if total > 0 else 0.0
    acc = 100. * correct / total if total > 0 else 0.0
    return avg_loss, acc

def train(args):
    # DDP Setup
    rank, world_size, local_rank, use_ddp = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if is_main_process():
        print(f"Using device: {device}")
        print(f"DDP: {'Enabled' if use_ddp else 'Disabled'} (World Size: {world_size})")
    
    # Class Map inverse for logging
    inv_map = {v: k for k, v in CLASS_MAP.items()}
    
    # 1. Train Dataset
    train_dataset = WheatDataset(root_dir=args.train_dir, transform=None, mode='train', return_name=True, expand_data=args.expand_data)
    clean_dataset = WheatDataset(root_dir=args.train_dir, transform=None, mode='all', return_name=True, expand_data=False)

    if is_main_process():
        print(f"Train Dataset Size: {len(train_dataset)}")
    
    # Validation subset
    total_samples = len(clean_dataset)
    val_size = min(600, total_samples // 2)
    np.random.seed(42)
    val_indices = np.random.choice(total_samples, val_size, replace=False)
    val_subset = torch.utils.data.Subset(clean_dataset, val_indices)
    
    # DDP Samplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if is_main_process():
        print(f"Validation Dataset Size: {len(val_subset)}")
    
    # Test Dataset
    test_dataset = WheatDataset(root_dir=args.test_dir, mode='all', return_name=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if is_main_process():
        print(f"Test Dataset Size: {len(test_dataset)}")

    # Model
    model = SpectralMambaClassifier(
        num_classes=3, 
        d_model=args.d_model, 
        depth=args.depth, 
        dropout=args.dropout, 
        drop_path_rate=args.drop_path,
        use_checkpoint=args.gradient_checkpoint  # Memory optimization
    )
    
    # Load Pretrained Weights if provided
    if args.pretrained:
        print(f"[INFO] Loading Pretrained Weights from {args.pretrained}")
        requested_path = args.pretrained
        migrated_path = requested_path.replace("mae_pretrain_best.pth", "mae_pretrain_last.pth")

        # Prefer LAST checkpoint path when an old BEST filename is passed.
        candidate_paths = []
        if migrated_path != requested_path:
            print(f"[INFO] Switched MAE preload target: BEST -> LAST ({migrated_path})")
            candidate_paths.append(migrated_path)
        candidate_paths.append(requested_path)

        search_paths = []
        seen = set()
        for p in candidate_paths:
            for c in (p, os.path.join("/kaggle/working/", os.path.basename(p))):
                if c not in seen:
                    seen.add(c)
                    search_paths.append(c)

        load_path = None
        for c in search_paths:
            if os.path.exists(c):
                load_path = c
                break

        if load_path and load_path != requested_path:
            print(f"[INFO] Resolved pretrained weights path: {load_path}")
        if load_path is None:
            print(f"[WARNING] Pretrained file not found. Tried: {search_paths}")

        if load_path:
            state_dict = torch.load(load_path, map_location=device)
            # The pretrained dict is from model.encoder, which IS SpectralMambaClassifier
            # However, the head might be different or uninitialized in pretrain.
            # We load strictly compatible keys.
            
            # Filter out head keys just in case (though MAE encoder doesn't train head, it has one)
            # Actually pretrain.py saves "model.encoder.state_dict()".
            # This matches SpectralMambaClassifier exactly.
            
            # Auto-fix DDP prefix mismatch (Before filtering!)
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            
            if len(model_keys & ckpt_keys) == 0:
                # No overlap - likely prefix mismatch
                sample_model_key = next(iter(model_keys))
                sample_ckpt_key = next(iter(ckpt_keys))
                
                if sample_model_key.startswith("module.") and not sample_ckpt_key.startswith("module."):
                    state_dict = {f"module.{k}": v for k, v in state_dict.items()}
                    print("[INFO] Added 'module.' prefix to checkpoint keys for DDP compatibility.")
                elif not sample_model_key.startswith("module.") and sample_ckpt_key.startswith("module."):
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                    print("[INFO] Stripped 'module.' prefix from checkpoint keys.")

            # Filter state dict to avoid shape mismatch (e.g. if the head dimension changed)
            model_dict = model.state_dict()
            
            # === CRITICAL DEBUG: Verify shape matching ===
            name_match = {k for k in state_dict if k in model_dict}
            shape_match = {k for k in name_match if state_dict[k].shape == model_dict[k].shape}
            shape_mismatch = {k for k in name_match if state_dict[k].shape != model_dict[k].shape}
            name_only_in_ckpt = {k for k in state_dict if k not in model_dict}
            
            print(f"[DEBUG PRETRAIN] Checkpoint keys: {len(state_dict)}, Model keys: {len(model_dict)}")
            print(f"[DEBUG PRETRAIN] Name match: {len(name_match)}, Shape match: {len(shape_match)}, Shape MISMATCH: {len(shape_mismatch)}")
            print(f"[DEBUG PRETRAIN] Keys only in checkpoint (not in model): {len(name_only_in_ckpt)}")
            
            if len(shape_mismatch) > 0:
                print(f"[WARNING] Shape mismatches found! First 5:")
                for k in list(shape_mismatch)[:5]:
                    print(f"  {k}: ckpt={state_dict[k].shape} vs model={model_dict[k].shape}")
            
            if len(shape_match) > 0:
                sample_key = next(iter(shape_match))
                print(f"[DEBUG PRETRAIN] Sample matching key: {sample_key}, shape={state_dict[sample_key].shape}")
            # === END DEBUG ===
            
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            msg = model.load_state_dict(filtered_dict, strict=False)
            print(f"[INFO] Pretrained weights loaded. Matching keys: {len(filtered_dict)}/{len(state_dict)}")
            print(f"[INFO] Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            if len(msg.missing_keys) > 0:
                print(f"[INFO] First 5 missing keys: {msg.missing_keys[:5]}")
            
            # Expected missing keys often include 'head.weight' if the model architecture was improved.
            # We should probably RESET the head to ensure we learn classification from scratch.
            
            print("[INFO] Resetting Classification Head for Fine-tuning (Xavier Init)...")
            # Ensure head is NOT randomly initialized (user request: su yi ji) -> Use Structured Xavier
            nn.init.xavier_uniform_(model.head.weight)
            if model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
            
        else:
            print(f"[WARNING] Pretrained file not found: {args.pretrained}")
    
    # LoRA Injection (Before freezing logic, before optimizer)
    if args.use_lora:
        print(f"[INFO] Injecting LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        inject_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        
    model = model.to(device)
    

    # T4 GPU (SM 7.5) does NOT have native BF16 Tensor Cores.
    # torch.cuda.is_bf16_supported() returns True but uses SOFTWARE EMULATION (extremely slow).
    # FORCE FP16 for T4 native Tensor Core acceleration.
    use_amp = args.amp and device.type == 'cuda'
    amp_dtype = torch.float16
    scaler = GradScaler(enabled=use_amp)
    
    if use_amp:
        if is_main_process():
            print(f"[INFO] AMP Enabled with dtype={amp_dtype}")
    
    # DDP Wrap
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process():
            print("[INFO] Model wrapped with DDP")
    
    # Freeze Logic (Initial)
    if args.freeze_epochs > 0:
        print(f"[INFO] Freezing Encoder for first {args.freeze_epochs} epochs...")
        # Freeze all except head AND LoRA parameters
        # LoRA params (lora_A, lora_B) should remain trainable even during freeze
        for name, param in model.named_parameters():
             if "head" in name or "lora_" in name:
                 param.requires_grad = True
             else:
                 param.requires_grad = False
        
        # Log trainable params during freeze
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Freeze Mode: Trainable params = {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
                         
        # Ensure dropout in head works, but backbone is deterministic
        # Use model.module for DDP compatibility
        base_model = model.module if use_ddp else model
        base_model.layers = base_model.layers.eval()
        base_model.norm_f = base_model.norm_f.eval()
        base_model.wavelength_sensing = base_model.wavelength_sensing.eval()
        base_model.pos_encoder = base_model.pos_encoder.eval()
        base_model.pos_proj = base_model.pos_proj.eval()
        base_model.patch_embed = base_model.patch_embed.eval()
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Layer-wise Learning Rate Logic
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.learning_rate * 0.1
    if is_main_process():
        print(f"[INFO] Learning Rates: Head={args.learning_rate}, Backbone={backbone_lr}")
    
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    # 2. Create Optimizer Group
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': args.learning_rate}
    ], momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    
    # Scheduler
    # Scheduler
    if args.epochs <= 5:
        # If very few epochs, just use Cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        # 5 epochs warmup
        # LinearLR starts from start_factor * lr -> lr over total_iters
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=5
        )
        # Main scheduler: Cosine Annealing
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - 5, eta_min=1e-6
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5]
        )
    
    best_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    # NOTE: scheduler.step() for warmup init is deferred to AFTER checkpoint resume.
    # See below after resume logic.
    
    # Gradient Accumulation Steps
    grad_accum_steps = args.grad_accum_steps
    eff_batch_size = args.batch_size * grad_accum_steps * world_size
    if is_main_process():
        print(f"Gradient Accumulation Steps: {grad_accum_steps}")
        print(f"Effective Batch Size: {eff_batch_size}")
    
    # --- Resume Logic (Auto-detect checkpoint_last.pth) ---
    if args.resume is None:
        possible_last = os.path.join(args.output_dir, "finetune_checkpoint_last.pth")
        if os.path.exists(possible_last):
            args.resume = possible_last
            if is_main_process():
                print(f"[INFO] Auto-resuming from FOUND last checkpoint: {args.resume}")
    
    if args.resume and os.path.exists(args.resume):
        if is_main_process():
            print(f"[INFO] Resuming fine-tuning from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Load Model State
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Auto-fix DDP prefix mismatch
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            
            if len(model_keys & ckpt_keys) == 0 and len(ckpt_keys) > 0:
                sample_model_key = next(iter(model_keys))
                sample_ckpt_key = next(iter(ckpt_keys))
                
                if sample_model_key.startswith("module.") and not sample_ckpt_key.startswith("module."):
                    state_dict = {f"module.{k}": v for k, v in state_dict.items()}
                    if is_main_process():
                        print("[INFO] Added 'module.' prefix to checkpoint keys for DDP compatibility.")
                elif not sample_model_key.startswith("module.") and sample_ckpt_key.startswith("module."):
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                    if is_main_process():
                        print("[INFO] Stripped 'module.' prefix from checkpoint keys.")
            
            msg = model.load_state_dict(state_dict, strict=False)
            if is_main_process():
                print(f"[INFO] Resumed model. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        
        # Load Optimizer & Scheduler
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        
        if is_main_process():
            print(f"  > Resumed at Epoch {start_epoch}, Best Loss: {best_loss:.4f}, Patience: {patience_counter}")
    else:
        # Only step scheduler for warmup init when NOT resuming from checkpoint.
        # When resuming, scheduler state is already restored from checkpoint.
        if args.epochs > 5:
            scheduler.step()
    
    # Re-apply freeze logic after resume (freeze state is not saved in checkpoint)
    if args.freeze_epochs > 0 and start_epoch < args.freeze_epochs:
        if is_main_process():
            print(f"[INFO] Re-applying freeze for epochs {start_epoch} to {args.freeze_epochs}...")
        for name, param in model.named_parameters():
            if "head" in name or "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        base_model = model.module if use_ddp else model
        base_model.layers = base_model.layers.eval()
        base_model.norm_f = base_model.norm_f.eval()
        base_model.wavelength_sensing = base_model.wavelength_sensing.eval()
        base_model.pos_encoder = base_model.pos_encoder.eval()
        base_model.pos_proj = base_model.pos_proj.eval()
        base_model.patch_embed = base_model.patch_embed.eval()
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler (important for shuffling in DDP)
        if use_ddp:
            train_sampler.set_epoch(epoch)
            
        # Unfreeze Logic
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            print(f"[INFO] Epoch {epoch}: Unfreezing Encoder (Mode: {'LoRA' if args.use_lora else 'Full'})!")
            
            if args.use_lora:
                # LoRA Mode: Only unfreeze LoRA params + Head
                # Original weights remain frozen
                for name, param in model.named_parameters():
                    if "lora_" in name or "head" in name:
                         param.requires_grad = True
                    else:
                         param.requires_grad = False
                print("[INFO] LoRA Adapters Unfrozen.")
            else:
                # Full Finetune Mode
                for param in model.parameters():
                    param.requires_grad = True
            
            # Reset to full train mode
            model.train()
                
        model.train()
        # Re-apply eval on frozen backbone modules to keep BatchNorm using running stats
        if args.freeze_epochs > 0 and epoch < args.freeze_epochs:
            base_model = model.module if use_ddp else model
            base_model.layers = base_model.layers.eval()
            base_model.norm_f = base_model.norm_f.eval()
            base_model.wavelength_sensing = base_model.wavelength_sensing.eval()
            base_model.pos_encoder = base_model.pos_encoder.eval()
            base_model.pos_proj = base_model.pos_proj.eval()
            base_model.patch_embed = base_model.patch_embed.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Augmentation Stats
        epoch_smote_samples = 0
        epoch_flipped_samples = 0
        
        # Store errors for this epoch
        epoch_errors = []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main_process())
        
        # Zero grad outside loop for first step
        optimizer.zero_grad()
        
        # Note: dataset now returns (data, label, filename)
        for i, (batch_data, batch_labels, batch_filenames) in enumerate(loop):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            # --- Spatial Augmentation (Flips) ---
            # Now handled by dataset.py expanded_data logic

            # DDP + Gradient Accumulation: use no_sync() on intermediate steps
            # Only sync gradients on the final accumulation step
            is_last_accum_step = (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader)
            maybe_no_sync = model.no_sync() if (use_ddp and not is_last_accum_step) else contextlib.nullcontext()
            
            with maybe_no_sync:
                # Forward with AMP autocast
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    
                    # Scale Loss for gradient accumulation
                    loss = loss / grad_accum_steps
                
                # Backward with scaler (for FP16) or regular backward (for BF16)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Step only every grad_accum_steps
            if is_last_accum_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Recover true loss for logging
            running_loss += loss.item() * grad_accum_steps
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            
            # Check correctness and log errors
            is_correct = (predicted == batch_labels)
            correct += is_correct.sum().item()
            
            # Log specific errors
            # batch_filenames is a tuple/list
            for j in range(len(batch_labels)):
                if not is_correct[j]:
                    true_cls = inv_map.get(batch_labels[j].item(), "Unknown")
                    pred_cls = inv_map.get(predicted[j].item(), "Unknown")
                    fname = batch_filenames[j]
                    epoch_errors.append({"filename": fname, "true": true_cls, "pred": pred_cls})
            
            loop.set_postfix(loss=loss.item() * grad_accum_steps, acc=100. * correct / (total + 1e-6))
            
        scheduler.step()
        
        # Avoid division by zero
        if total == 0:
            train_acc = 0.0
        else:
            train_acc = 100. * correct / total
            
        avg_loss = running_loss / len(train_loader)
        
        # --- Validation Step (Clean Subset) ---
        # --- Validation Step (Clean Subset) ---
        # Enable TTA for Validation Metrics too if requested
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, tta=args.tta)
        
        if is_main_process():
            print(f"Epoch {epoch+1} Summary: Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save errors to CSV and Print Summary
        if len(epoch_errors) > 0:
            # Save to CSV
            err_df = pd.DataFrame(epoch_errors)
            # Sort by filename
            err_df = err_df.sort_values("filename")
            err_path = os.path.join(args.output_dir, f"train_errors_epoch_{epoch+1}.csv")
            err_df.to_csv(err_path, index=False)
            print(f"--- Saved {len(epoch_errors)} errors to {err_path} ---")
            
            # Print Summary Counts instead of all lines
            err_counts = err_df.groupby(['true', 'pred']).size().reset_index(name='count')
            print(f"--- Error Summary (Epoch {epoch+1}) ---")
            print(err_counts.to_string(index=False))
            print("------------------------------------------------")
        
        # Save model if Clean Val Loss improves (only on main process)
        if is_main_process():
            if val_loss < best_loss:
                best_loss = val_loss
                model_to_save = model.module if use_ddp else model
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "best_loss_model.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Early Stopping Counter: {patience_counter}/{args.patience}")
        
        # Sync patience_counter across all processes for consistent early stopping
        if use_ddp:
            patience_tensor = torch.tensor([patience_counter], device=device)
            dist.broadcast(patience_tensor, src=0)
            patience_counter = patience_tensor.item()
            
        if patience_counter >= args.patience:
            if is_main_process():
                print(f"Early stop at epoch {epoch+1}")
            break
        
        # Save Checkpoint for Resume (only on main process)
        if is_main_process():
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': (model.module if use_ddp else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'best_loss': best_loss,
                'patience_counter': patience_counter,
            }
            torch.save(checkpoint_dict, os.path.join(args.output_dir, "finetune_checkpoint_last.pth"))
            
        # Inference every epoch (only on main process)
        if is_main_process():
            inference_loop(model.module if use_ddp else model, test_loader, device, epoch, args.output_dir, tta=args.tta)
    
    # Cleanup DDP
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated default paths based on user request
    parser.add_argument("--train_dir", type=str, default="/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/train/HS")
    parser.add_argument("--test_dir", type=str, default="/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared/val/HS")
    
    # Backwards compatibility names if needed (but we use train_dir/test_dir now)
    parser.add_argument("--data_dir", type=str, help="Deprecated", default="")
    
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    # val_split not used anymore
    parser.add_argument("--val_split", type=float, default=0.0) 
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient Accumulation Steps")
    parser.add_argument("--pretrained", type=str, default="", help="Path to MAE pretrained encoder weights")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability for Classification Head")
    parser.add_argument("--drop_path", type=float, default=0.1, help="Drop Path rate (Stochastic Depth)")
    parser.add_argument("--backbone_lr", type=float, default=None, help="Learning rate for Backbone (default: 0.1 * lr)")
    parser.add_argument("--freeze_epochs", type=int, default=0, help="Number of epochs to freeze encoder")
    parser.add_argument("--expand_data", type=str, nargs='?', const='True', default="False", help="Enable augmentation: 'True' (Stage 1), 'stage2' (Stage 2), or 'False'")
    parser.add_argument("--tta", action='store_true', default=True, help="Enable Test Time Augmentation (TTA)")
    parser.add_argument("--no_tta", action='store_false', dest='tta', help="Disable Test Time Augmentation (TTA)")
    parser.add_argument("--resume", type=str, default=None, help="Path to fine-tune checkpoint to resume from (auto-detects finetune_checkpoint_last.pth)")
    
    # LoRA Utils (Conservative defaults for limited fine-tuning data)
    parser.add_argument("--use_lora", action='store_true', help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA Rank (smaller=fewer params, safer for small datasets)")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA Alpha (alpha/r controls scaling)")
    
    # Acceleration Options
    parser.add_argument("--amp", action='store_true', default=True, help="Enable AMP mixed precision training")
    parser.add_argument("--no_amp", action='store_false', dest='amp', help="Disable AMP")

    # Memory Optimization
    parser.add_argument("--gradient_checkpoint", action='store_true', default=True, help="Enable gradient checkpointing (saves ~30-40%% VRAM)")
    parser.add_argument("--no_gradient_checkpoint", action='store_false', dest='gradient_checkpoint', help="Disable gradient checkpointing")
    
    args = parser.parse_args()

    # In Kaggle, always write outputs directly to root for easier manual download.
    kaggle_root = "/kaggle/working"
    if os.path.isdir(kaggle_root):
        if os.path.abspath(args.output_dir) != os.path.abspath(kaggle_root):
            print(f"[INFO] Override output_dir to Kaggle root: {kaggle_root}")
        args.output_dir = kaggle_root
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train(args)
