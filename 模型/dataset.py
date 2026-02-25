
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from consts import SELECTED_BANDS, IDX_BLUE, IDX_GREEN, IDX_RED, IDX_RED_EDGE, IDX_NIR, CLASS_MAP, MAX_PIXEL_VAL, BLACKLIST_FILES

# Guard against occasional corrupted preprocessed patches (hot/dead pixels).
PRETRAIN_PATCH_CLIP_RANGE = (-0.1, 1.5)

# Try to import slic
try:
    from skimage.segmentation import slic
    HAS_SKIMAGE = True
except ImportError:
    print("[WARNING] scikit-image not found. Superpixel CutMix will fallback to identity/SMOTE.")
    HAS_SKIMAGE = False

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    print("[WARNING] scipy not found. S-G smoothing will be disabled.")
    HAS_SCIPY = False

class WheatDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='all', return_name=False, expand_data=False):
        """
        Args:
            root_dir (string): Directory with all the .tif images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train' (exclude val_), 'test' (only val_), 'all' (no filter).
            return_name (bool): If True, returns (data, label, filename).
            expand_data (bool): If True, doubles the dataset length. 
                                First half = Spatial Aug (Flip), Second half = SMOTE & CutMix.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.return_name = return_name
        self.expand_data = expand_data
        self.aug_version = 1 # Default Stage 1
        
        if isinstance(expand_data, str):
            if expand_data.lower() == "true":
                self.expand_data = True
            elif expand_data.lower() == "false":
                self.expand_data = False
            elif expand_data.startswith("stage"):
                self.aug_version = int(expand_data.replace("stage", ""))
                self.expand_data = True
            else:
                self.expand_data = False # Default or handle other cases
        
        # Glob all .tif and .pt files
        all_paths = glob.glob(os.path.join(root_dir, "*.tif"))
        if len(all_paths) == 0:
             # Try case-insensitive extension manually
             all_paths = glob.glob(os.path.join(root_dir, "*.[tT][iI][fF]"))

        # --- Apply User Blacklist ---
        original_count = len(all_paths)
        all_paths = [p for p in all_paths if os.path.basename(p) not in BLACKLIST_FILES]
        filtered_count = original_count - len(all_paths)
        if filtered_count > 0:
            print(f"[INFO] Filtered {filtered_count} blacklisted files from {root_dir}")
        
        # Also look for .pt files (Generated Patches)
        pt_paths = glob.glob(os.path.join(root_dir, "*.pt"))
        if len(pt_paths) > 0:
            print(f"[INFO] Found {len(pt_paths)} .pt files (Generated Patches).")
            # Filter out macOS ._ files just in case
            pt_paths = [p for p in pt_paths if not os.path.basename(p).startswith("._")]
            all_paths.extend(pt_paths)
        
        print(f"[DEBUG] Found {len(all_paths)} files in {root_dir}")
        if len(all_paths) > 0:
            print(f"[DEBUG] First 5 files: {[os.path.basename(p) for p in all_paths[:5]]}")
        
        # Filter based on mode
        self.image_paths = []
        for p in all_paths:
            fname = os.path.basename(p)
            is_val = fname.lower().startswith("val_")
            
            if mode == 'train':
                if not is_val:
                    self.image_paths.append(p)
            elif mode == 'test':
                if is_val:
                    self.image_paths.append(p)
            else:
                self.image_paths.append(p)

        if len(self.image_paths) == 0:
            print(f"Warning: No valid files found in {root_dir} with mode={mode}")
            
        # Build Index Map for SMOTE/CutMix if expanded
        self.indices_by_class = {}
        if self.expand_data:
            print("[INFO] Dataset Expansion Enabled: Building Class Index Map...")
            for idx, p in enumerate(self.image_paths):
                fname = os.path.basename(p)
                label_str = fname.split('_')[0]
                label_int = CLASS_MAP.get(label_str, CLASS_MAP['Other'])
                
                if label_int not in self.indices_by_class:
                    self.indices_by_class[label_int] = []
                self.indices_by_class[label_int].append(idx)
            print(f"[INFO] Class Distribution: { {k: len(v) for k,v in self.indices_by_class.items()} }")

    def __len__(self):
        if self.expand_data:
            if self.aug_version == 3:
                return 10 * len(self.image_paths)
            if self.aug_version == 4:
                return len(self.image_paths) # 1x for convergence
            return 2 * len(self.image_paths)
        return len(self.image_paths)

    def _compute_spectral_tensor(self, image, savgol_window=None, is_pre_normalized=False):
        """Processes raw image array into (116, H, W) tensor with VIs and optional S-G smoothing"""
        # image: (C, H, W) raw numpy, already transposed if needed
        image = image.astype(np.float32)

        # 1. Normalize (Skip if already 0-1)
        if not is_pre_normalized:
            # Consistent Robust Normalization (Same as prepare_whu_data.py)
            # Clip 2%-98% to ignore outliers
            p_low, p_high = np.percentile(image, [2, 98])
            image = np.clip(image, p_low, p_high) # keep dtype or cast
            
            if p_high > p_low:
                image = (image - p_low) / (p_high - p_low)
            else:
                image = image / MAX_PIXEL_VAL # Fallback
            
            image = image.astype(np.float32) # Ensure final float32
        
        # 2. Optional S-G Smoothing (Apply to 110 spectral bands before VI)
        if savgol_window is not None and HAS_SCIPY:
            # savgol_filter expects signal on an axis. image is (C, H, W).
            # We want to smooth along C.
            # (C, H, W) -> (H, W, C) for easier filtering row by row, or just use axis=0
            # polyorder=2 is standard for S-G
            image[:110, :, :] = savgol_filter(image[:110, :, :], window_length=savgol_window, polyorder=2, axis=0)

        # 3. Calculate VIs
        eps = 1e-8
        blue = image[IDX_BLUE, :, :]
        green = image[IDX_GREEN, :, :]
        red = image[IDX_RED, :, :]
        red_edge = image[IDX_RED_EDGE, :, :]
        nir = image[IDX_NIR, :, :]

        ndvi = (nir - red) / (nir + red + eps)
        ndre = (nir - red_edge) / (nir + red_edge + eps)
        evi = 2.5 * (nir - red) / (nir + 2.4 * red + 1 + eps)
        mtci = (nir - red_edge) / (red_edge - red + eps)
        ci_rededge = (nir / (red_edge + eps)) - 1
        gndvi = (nir - green) / (nir + green + eps)
        psri = (red - blue) / (red_edge + eps)
        sipi = (nir - blue) / (nir - red + eps)
        mcari = ((red_edge - red) - 0.2 * (red_edge - green)) * (red_edge / (red + eps))

        # Stack VIs
        vis = np.stack([ndvi, ndre, gndvi, psri, sipi, mcari, evi, mtci, ci_rededge], axis=0) 
        
        # FIX: Clamp VIs to [-5, 5] to preserve valid high values (EVI ~2.5) while killing outliers (5000)
        vis = np.clip(vis, -5.0, 5.0)
        
        # 4. Prune Bands
        spectral_data = image[SELECTED_BANDS, :, :]
        
        # 5. Concatenate
        combined_data = np.concatenate([spectral_data, vis], axis=0) 
        
        # DEBUG: Check range
        # if torch.rand(1) < 0.01: # 1% chance
        #     print(f"[DATASET] Value check: Min={combined_data.min():.4f}, Max={combined_data.max():.4f}, Mean={combined_data.mean():.4f}, NormFactor={MAX_PIXEL_VAL}")

        return torch.from_numpy(combined_data).float()

    def _load_single(self, idx, savgol_window=None):
        """Helper to load a single image and label by index"""
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        
        try:
            if img_path.endswith('.pt'):
                # Handle PyTorch Tensor (Pre-processed Patch)
                # Assuming shape (C, H, W) and already normalized 0-1
                image_tensor = torch.load(img_path)
                image_np = image_tensor.numpy() 
                is_pre_normalized = True
                
                # Check shape, .pt might be (125, 32, 32)
                image = np.clip(image_np, PRETRAIN_PATCH_CLIP_RANGE[0], PRETRAIN_PATCH_CLIP_RANGE[1]).astype(np.float32, copy=False)
            else:
                # Handle Tiff (Original S185)
                image = tifffile.imread(img_path)
                is_pre_normalized = False

        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return None, None, filename

        # Ensure (C, H, W)
        if image.shape[0] in [125, 126]:
             pass
        elif image.shape[-1] in [125, 126]:
            image = image.transpose(2, 0, 1)
        
        data_tensor = self._compute_spectral_tensor(image, savgol_window=savgol_window, is_pre_normalized=is_pre_normalized)
        
        label_str = filename.split('_')[0] 
        label = CLASS_MAP.get(label_str, CLASS_MAP['Other']) 
        return data_tensor, label, filename

    def _get_smote_sample(self, real_idx, data, label):
        """Pure Same-Class SMOTE"""
        candidates = self.indices_by_class.get(label, [real_idx])
        idx2 = np.random.choice(candidates) if len(candidates) > 1 else real_idx
        data2, _, _ = self._load_single(idx2)
        
        if data2 is not None:
            lam = np.random.beta(1.0, 1.0)
            data = lam * data + (1 - lam) * data2
        return data

    def _get_cutmix_sample(self, real_idx, data, label):
        """Pure Superpixel CutMix"""
        if not HAS_SKIMAGE:
            return self._get_smote_sample(real_idx, data, label)
            
        candidates = self.indices_by_class.get(label, [real_idx])
        idx2 = np.random.choice(candidates) if len(candidates) > 1 else real_idx
        data2, _, _ = self._load_single(idx2)
        
        if data2 is not None:
            try:
                img_np = data.numpy()
                img_rgb = img_np[[IDX_RED, IDX_GREEN, IDX_BLUE], :, :].transpose(1, 2, 0)
                segments = slic(img_rgb, n_segments=50, compactness=10, sigma=1, start_label=1)
                unique_segs = np.unique(segments)
                num_segs_to_take = max(1, len(unique_segs) // 2)
                chosen_segs = np.random.choice(unique_segs, num_segs_to_take, replace=False)
                mask = np.isin(segments, chosen_segs)
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
                data = data * (1 - mask_tensor) + data2 * mask_tensor
            except Exception:
                # Fallback to SMOTE
                lam = np.random.beta(1.0, 1.0)
                data = lam * data + (1 - lam) * data2
        return data

    def _get_expanded_sample(self, idx, real_len, real_idx, data, label):
        """Augmentation Logic for idx >= real_len"""
        
        # Part 1: Spatial Augmentation (Pure Spatial, No Mixing)
        # Flip half of the expanded data (but current logic is split by index range)
        # Original req: First half = Spatial Aug (Flip). Wait, idx < real_len is ORIGINAL data?
        # Typically Dataset[0:N] is original. Dataset[N:2N] is Augmented.
        # But User request: "One epoch contains 600 spatial + 600 smote".
        # This implies NO original data? Or "Spatial Aug" IS the original data representation?
        # Safe bet: 0:N -> Spatial Aug (Original + Flip). N:2N -> Mixed.
        
        # Actually, let's follow the standard "Expanded" pattern:
        # 0 ~ N-1: Spatial Aug (Flip/Rotate or Identity).
        # N ~ 2N-1: Mixed (SMOTE/CutMix).
        
        # This means 0-N-1 is effectively the "Standard" batch but augmented.
        
        # Logic for N ~ 2N-1 (The "Extra" 600 samples)
        
        # Random choice: 50% SMOTE, 50% CutMix
        use_cutmix = torch.rand(1) < 0.5
        
        if HAS_SKIMAGE and use_cutmix:
            # --- Superpixel CutMix ---
            # 1. Find partner
            candidates = self.indices_by_class.get(label, [real_idx])
            idx2 = np.random.choice(candidates) if len(candidates) > 1 else real_idx
            data2, _, _ = self._load_single(idx2)
            
            if data2 is not None:
                C, H, W = data.shape
                
                # SLIC is slow on 116 channels. Use RGB indices (110-116 are VIs, need to check where RGB is in combined)
                # combined_data structure: [0-109 spectral, 110-118 VIs]
                # RGB indices in spectral: IDX_RED=52, IDX_GREEN=25, IDX_BLUE=5.
                
                try:
                    img_np = data.numpy() # (C, H, W)
                    img_rgb = img_np[[IDX_RED, IDX_GREEN, IDX_BLUE], :, :].transpose(1, 2, 0) # (H, W, 3)
                    
                    # Normalize for SLIC? SLIC works on image structure.
                    # It expects (H, W, C) or (H, W).
                    segments = slic(img_rgb, n_segments=50, compactness=10, sigma=1, start_label=1)
                    
                    # Select random segments to paste from B (data2) to A (data1)
                    # Unique segments
                    unique_segs = np.unique(segments)
                    # Pick random subset (e.g. 30-50% of area)
                    num_segs_to_take = max(1, len(unique_segs) // 2)
                    chosen_segs = np.random.choice(unique_segs, num_segs_to_take, replace=False)
                    
                    mask = np.isin(segments, chosen_segs) # Boolean mask (H, W)
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) # (1, H, W)
                    
                    # Paste B onto A
                    # New = A * (1-Mask) + B * Mask
                    data = data * (1 - mask_tensor) + data2 * mask_tensor
                    
                except Exception as e:
                    # Fallback to SMOTE if SLIC fails
                    lam = np.random.beta(1.0, 1.0)
                    data = lam * data + (1 - lam) * data2
                    
        else:
            # --- Same-Class SMOTE ---
            candidates = self.indices_by_class.get(label, [real_idx])
            idx2 = np.random.choice(candidates) if len(candidates) > 1 else real_idx
            data2, _, _ = self._load_single(idx2)
            
            if data2 is not None:
                lam = np.random.beta(1.0, 1.0)
                data = lam * data + (1 - lam) * data2
                
        return data

    def __getitem__(self, idx):
        real_len = len(self.image_paths)
        real_idx = idx % real_len
        
        # Load Base Image
        data, label, fname = self._load_single(real_idx)
        if data is None:
            return torch.zeros(119, 32, 32), 0, "error"
            
        if self.expand_data:
            if self.aug_version == 1:
                # Stage 1: Original/Flip + Random Mix
                if idx < real_len:
                    if torch.rand(1) < 0.5: data = torch.flip(data, [1])
                    if torch.rand(1) < 0.5: data = torch.flip(data, [2])
                else:
                    data = self._get_expanded_sample(idx, real_len, real_idx, data, label)
            elif self.aug_version == 2:
                # Stage 2: Whole dataset is Mixed, 50% SMOTE, 50% CutMix, NO FLIP
                # idx < real_len -> SMOTE, idx >= real_len -> CutMix
                if idx < real_len:
                    data = self._get_smote_sample(real_idx, data, label)
                else:
                    data = self._get_cutmix_sample(real_idx, data, label)
            elif self.aug_version == 3:
                # Stage 3: 10x Expansion
                # 0-3: Flips (Original, H, V, HV)
                # 4-7: Mixed (SMOTE x2, CutMix x2)
                # 8-9: Spectral (S-G W=5, S-G W=9)
                
                group = idx // real_len
                
                if group == 0: # Original
                    pass 
                elif group == 1: # HFlip
                    data = torch.flip(data, [1])
                elif group == 2: # VFlip
                    data = torch.flip(data, [2])
                elif group == 3: # HVFlip
                    data = torch.flip(data, [1, 2])
                elif group in [4, 5]: # SMOTE (2x)
                    data = self._get_smote_sample(real_idx, data, label)
                elif group in [6, 7]: # CutMix (2x)
                    data = self._get_cutmix_sample(real_idx, data, label)
                elif group == 8: # S-G Smooth W=5
                    # We need to reload to apply S-G on raw 110 bands
                    data, _, _ = self._load_single(real_idx, savgol_window=5)
                elif group == 9: # S-G Smooth W=9
                    data, _, _ = self._load_single(real_idx, savgol_window=9)
            elif self.aug_version == 4:
                # Stage 4: 1x Dataset with Random Flips (Pure Convergence)
                # Standard Spatial Augmentation only
                if torch.rand(1) < 0.5: data = torch.flip(data, [1])
                if torch.rand(1) < 0.5: data = torch.flip(data, [2])
                    
        # 6. Transform
        if self.transform:
            data = self.transform(data)
            
        if self.return_name:
            return data, label, fname
            
        return data, label

if __name__ == "__main__":
    try:
        ds = WheatDataset(root_dir=r"C:\Users\feng\Desktop\kaggle", expand_data=True) 
        print(f"Found {len(ds)} images (Expanded).")
        # Test one fetch
        d, l = ds[len(ds)-1] # Test expansion
        print(f"Sample shape: {d.shape}")
    except Exception as e:
        print(f"Test failed: {e}")
