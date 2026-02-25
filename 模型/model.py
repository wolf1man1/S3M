
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# IMPORTANT: Real Mamba2 from mamba_ssm is REQUIRED.
# DO NOT use any fallback - the fallback was accidentally overriding the real Mamba2!
from mamba_ssm import Mamba2
from torch.utils.checkpoint import checkpoint
print("[DEBUG] SUCCESS: Imported real 'Mamba2' from mamba_ssm package.")
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = alpha
        self.scaling = alpha / r
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(linear_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, linear_layer.out_features))
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    @property
    def weight(self):
        """Proxy to underlying linear weight for Mamba2 compatibility."""
        return self.linear.weight
    
    @property
    def bias(self):
        """Proxy to underlying linear bias for Mamba2 compatibility."""
        return self.linear.bias
    
    @property
    def in_features(self):
        return self.linear.in_features
    
    @property
    def out_features(self):
        return self.linear.out_features
            
    def forward(self, x):
        # Original output
        out = self.linear(x)
        # LoRA path
        lora_out = (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return out + lora_out

def inject_lora(model, r=4, alpha=8, target_modules=None, parent_name=""):
    """
    Replaces nn.Linear layers with LoRALinear recursively.
    
    CONSERVATIVE DEFAULTS for fine-tuning with limited data:
    - r=4 (smaller rank to minimize trainable params)
    - alpha=8 (alpha/r = 2, moderate scaling)
    - Only targets Mamba-related layers by default
    
    Args:
        model: The model to inject LoRA into
        r: LoRA rank (smaller = fewer params, less risk of overfitting)
        alpha: LoRA alpha scaling factor
        target_modules: List of substrings to match layer names. 
                       If None, defaults to Mamba-related layers only.
        parent_name: Internal use for tracking full path
    """
    # Conservative default: Only target last 3 layers of each stream
    # Layers 6, 7, 8 (0-indexed: 5, 6, 7) — preserves early feature extraction
    if target_modules is None:
        target_modules = [
            "spectral_layers.5", "spectral_layers.6", "spectral_layers.7",
            "spatial_layers.5", "spatial_layers.6", "spatial_layers.7",
        ]
    
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        # Recurse into children
        if len(list(module.children())) > 0:
            inject_lora(module, r, alpha, target_modules, full_name)
        
        if isinstance(module, nn.Linear):
            # Skip classification heads (these should be trained directly)
            if "head" in name.lower() or "pred_head" in name.lower():
                continue
            
            # Skip adapters and embeddings (these are task-specific anyway)
            skip_keywords = ["adapter", "embed", "pos_proj", "mlp", "ffn"]
            if any(kw in full_name.lower() for kw in skip_keywords):
                continue
            
            # Check if this layer matches any target module pattern
            should_inject = any(target in full_name.lower() for target in target_modules)
            
            if should_inject:
                new_module = LoRALinear(module, r=r, alpha=alpha)
                setattr(model, name, new_module)
                print(f"[LoRA] Injected: {full_name} (r={r}, alpha={alpha})")



from consts import WAVELENGTHS, SELECTED_BANDS, IDX_BLUE, IDX_GREEN, IDX_RED, IDX_RED_EDGE, IDX_NIR

class GaussianFourierMapping(nn.Module):
    """
    Random Gaussian Fourier Mapping.
    Maps physical wavelengths to high-dimensional embeddings.
    gamma(lambda) = [cos(2*pi*B*lambda), sin(2*pi*B*lambda)]
    """
    def __init__(self, input_dim=1, mapping_size=256, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        # B is fixed matrix sampled from N(0, sigma^2)
        # input_dim is usually 1 (wavelength scalar)
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, x):
        # x: (N, 1) or (1, C, 1) etc.
        # 2 * pi * B * x
        if x.dim() == 1:
            x = x.unsqueeze(1) # (C, 1)
        
        # (C, 1) @ (1, M) -> (C, M)
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) # (C, 2M)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor



class ImplicitWavelengthSensing(nn.Module):
    """
    Channel Attention mechanism that uses physical wavelength embeddings 
    to reweight the input channels.
    """
    def __init__(self, num_channels, wavelengths, embedding_dim=64):
        super().__init__()
        self.num_channels = num_channels
        
        # Mapping for Wavelengths
        self.fourier_mapper = GaussianFourierMapping(input_dim=1, mapping_size=embedding_dim // 2, scale=1.0)
        
        # Transform embeddings to attention weights
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Normalized Wavelengths (0-1 roughly)
        w_tensor = torch.tensor(wavelengths, dtype=torch.float32)
        min_w, max_w = w_tensor.min(), w_tensor.max()
        self.w_norm = (w_tensor - min_w) / (max_w - min_w + 1e-6)
        
        # Learnable embeddings for VIs
        # Calculate how many VIs we have
        self.num_vis = num_channels - len(wavelengths)
        self.vi_embeddings = nn.Parameter(torch.randn(self.num_vis, embedding_dim))
        
    def forward(self, x):
        # x: (Batch, Channels, H, W)
        # We want to reweight Channels.
        
        # 1. Get Spectral Embeddings (110, D)
        w_emb = self.fourier_mapper(self.w_norm.to(x.device)) # (110, D)
        
        # 2. Get VI Embeddings (num_vis, D)
        vi_emb = self.vi_embeddings # (num_vis, D)
        
        # 3. Concatenate (110 + num_vis, D)
        full_emb = torch.cat([w_emb, vi_emb], dim=0)
        
        # 4. Compute Attention Weights
        attn = self.mlp(full_emb) # (116, 1)
        attn = attn.view(1, -1, 1, 1) # (1, 116, 1, 1)
        
        return x * attn

# HybridManifoldPhysicalScanning Removed (User Request: Pure Spectral Scanning)

class PatchEmbed(nn.Module):
    """
    Simple 4x4 non-overlapping patch embedding.
    Input:  (B*C, 1, H, W)
    Output: (B*C, H/4, W/4, d_model)
    """
    def __init__(self, in_channels=1, d_model=512, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B*C, 1, H, W)
        x = self.proj(x)  # (B*C, d_model, H/4, W/4)
        # Keep NHWC after LayerNorm to avoid an extra permute back to NCHW.
        x = x.permute(0, 2, 3, 1)  # (B*C, H', W', D)
        x = self.norm(x)            # (B*C, H', W', D)
        return x

class DynamicSpectralFusion(nn.Module):
    """
    Content-Aware Spectral Weighting (SE-Block variant).
    Dynamically predicts per-image, per-position spectral weights.
    Squeeze: global context via band-averaging.
    Excitation: MLP predicts importance of each band.
    Reweight: weighted sum over bands.
    """
    def __init__(self, d_model, num_bands=119, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // reduction),
            nn.ReLU(),
            nn.Linear(d_model // reduction, num_bands),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (B, S, C, D)
        B, S, C, D = x.shape

        # 1. Squeeze: average over bands -> (B, S, D)
        context = x.mean(dim=2)

        # 2. Excitation: predict per-band weights -> (B, S, C)
        weights = self.gate(context)

        # 3. Reweight: weighted sum -> (B, S, D)
        weights = weights.view(B, S, C, 1)
        x_fused = (x * weights).sum(dim=2)

        return x_fused

class SpatialMambaBlock(nn.Module):
    """
    S3M Component: Spatial Scanning Mamba Block.
    Performs Mamba scan over the SPATIAL dimension (H*W) instead of Spectral.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        # x: (B, S, C, D)
        # B: Batch Size
        # S: Spatial Sequence Length (H*W)
        # C: Channel/Spectral Count
        # D: Feature Dimension
        
        B, S, C, D = x.shape
        
        # Target for Mamba: (Batch_Size, Seq_Len, Dim)
        # We want to scan S. So Batch should be B*C.
        
        # 1. Permute to (B, C, S, D) -> (B*C, S, D)
        x_in = x.permute(0, 2, 1, 3).contiguous().view(B*C, S, D)
        
        # 2. Apply Mamba
        # Mamba requires strict stride alignment. LayerNorm should be contiguous, but we enforce it.
        x_norm = self.norm(x_in).contiguous()
        x_out = self.mamba(x_norm)
        
        # 3. Reshape back to (B, S, C, D)
        x_out = x_out.view(B, C, S, D).permute(0, 2, 1, 3)
        
        # Residual connection
        return x + self.drop_path(x_out)





class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    Uses fewer KV heads than query heads for efficiency.
    Automatically uses Flash Attention via F.scaled_dot_product_attention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of query heads (default 8)
        num_kv_heads: Number of KV heads (default 4, GQA-4)
    """
    def __init__(self, d_model, num_heads=8, num_kv_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_groups = num_heads // num_kv_heads  # queries per KV head
        
        # Q uses full heads, K/V use fewer heads
        self.q_proj = nn.Linear(d_model, d_model)  # num_heads * head_dim
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, need_weights=False):
        B, S_q, _ = query.shape
        S_kv = key.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query).view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S_kv, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Expand KV heads for GQA: (B, num_kv_heads, S, D) -> (B, num_heads, S, D)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Flash Attention via PyTorch SDPA (automatically selects FlashAttn/MemEfficient)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        return self.out_proj(attn_out), None


class SpectralMambaBlock(nn.Module):
    """
    Pure Spectral Mamba block. Scans along channel dimension C.
    Input/Output: (B, S, C, D)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        # x: (B, S, C, D)
        B, S, C, D = x.shape
        x_flat = x.contiguous().view(B*S, C, D)
        x_norm = self.norm(x_flat)
        with torch.amp.autocast('cuda', enabled=False):
            delta = self.mamba(x_norm.float())
        delta = delta.to(x.dtype).view(B, S, C, D)
        return x + self.drop_path(delta)


class SpatialTransformerBlock(nn.Module):
    """
    Pure Spatial Transformer block. Self-Attention + FFN across spatial positions S.
    Input/Output: (B, S, C, D)
    """
    def __init__(self, d_model, num_heads=4, drop_path=0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_heads // 2)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.shared_attn = True # Optimization: Compute Spatial Attn on spectral mean
    
    
    
    def forward(self, x):
        # x: (B, S, C, D)
        B, S, C, D = x.shape
        
        if self.shared_attn:
            # OPTIMIZATION: Shared Spatial Attention
            # Compute attention map on the MEAN of spectral bands (Spatial Structure is shared)
            # (B, S, C, D) -> (B, S, D)
            x_mean = x.mean(dim=2)
            
            # Self-Attention on Mean
            x_norm = self.norm_attn(x_mean)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False) # (B, S, D)
            
            # Broadcast back to C
            # (B, S, D) -> (B, S, 1, D)
            attn_out = attn_out.unsqueeze(2)
            x = x + self.drop_path(attn_out) # Broadcast addition
            
            # Per-Band MLP (Features are distinct)
            # (B, S, C, D) -> (B*C, S, D)
            x_flat = x.permute(0, 2, 1, 3).contiguous().view(B*C, S, D)
            ffn_out = self.mlp(self.norm_mlp(x_flat))
            x_flat = x_flat + self.drop_path(ffn_out)
            
            return x_flat.view(B, C, S, D).permute(0, 2, 1, 3)
            
        else:
            # Original: Independent Attention per Band
            # Reshape: (B, S, C, D) -> (B*C, S, D)
            x_flat = x.permute(0, 2, 1, 3).contiguous().view(B*C, S, D)
            
            # Self-Attention (Pre-Norm + Internal Residual)
            x_norm = self.norm_attn(x_flat)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            x_flat = x_flat + self.drop_path(attn_out)
            
            # FFN (Pre-Norm + Internal Residual)
            ffn_out = self.mlp(self.norm_mlp(x_flat))
            x_flat = x_flat + self.drop_path(ffn_out)
            
            # Reshape back: (B*C, S, D) -> (B, C, S, D) -> (B, S, C, D)
            return x_flat.view(B, C, S, D).permute(0, 2, 1, 3)


class BottleneckFusionBlock(nn.Module):
    """
    Bottleneck Fusion between Spectral and Spatial streams.
    Projects both streams to a shared low-dimensional semantic space,
    fuses via add + MLP, then projects back.
    Forces the model to learn "which spatial textures correspond to which spectral curves".
    """
    def __init__(self, d_model, bottleneck_ratio=4):
        super().__init__()
        d_bottleneck = d_model // bottleneck_ratio
        
        # Projection down to bottleneck
        self.norm_spec = nn.LayerNorm(d_model)
        self.norm_spat = nn.LayerNorm(d_model)
        self.proj_spec_down = nn.Linear(d_model, d_bottleneck)
        self.proj_spat_down = nn.Linear(d_model, d_bottleneck)
        
        # Shared fusion MLP in bottleneck space
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_bottleneck, d_bottleneck),
            nn.GELU(),
            nn.Linear(d_bottleneck, d_bottleneck),
        )
        self.norm_fusion = nn.LayerNorm(d_bottleneck)
        
        # Projection back up to d_model (separate for each stream)
        self.proj_spec_up = nn.Linear(d_bottleneck, d_model)
        self.proj_spat_up = nn.Linear(d_bottleneck, d_model)
        
        # Static Learnable Gates
        self.gate_spec = nn.Parameter(torch.zeros(1))
        self.gate_spat = nn.Parameter(torch.zeros(1))
    
    def forward(self, x_spec, x_spat):
        # x_spec, x_spat: (B, S, C, D)
        B, S, C, D = x_spec.shape
        
        # Flatten to (B*S*C, D) for efficient linear ops
        spec_flat = self.norm_spec(x_spec).reshape(-1, D)
        spat_flat = self.norm_spat(x_spat).reshape(-1, D)
        
        # Project down to bottleneck
        spec_bn = self.proj_spec_down(spec_flat)  # (N, d_bottleneck)
        spat_bn = self.proj_spat_down(spat_flat)  # (N, d_bottleneck)
        
        # Fuse in shared semantic space
        fused = spec_bn + spat_bn  # element-wise fusion
        fused = fused + self.fusion_mlp(self.norm_fusion(fused))  # refine with MLP
        
        # Project back up (different projections for each stream)
        delta_spec = self.proj_spec_up(fused).view(B, S, C, D)
        delta_spat = self.proj_spat_up(fused).view(B, S, C, D)
        
        return x_spec + self.gate_spec.sigmoid() * delta_spec, \
               x_spat + self.gate_spat.sigmoid() * delta_spat


class CrossFusionBlock(nn.Module):
    """
    Bidirectional Cross-Attention fusion between Spectral and Spatial streams.
    Uses GQA for efficiency. Intended for deep layers where rich fusion is needed.
    Per-channel spatial cross-attention: (B*C, S, D) — efficient (S=64 tokens only).
    """
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.norm_spec_q = nn.LayerNorm(d_model)
        self.norm_spec_kv = nn.LayerNorm(d_model)
        self.norm_spat_q = nn.LayerNorm(d_model)
        self.norm_spat_kv = nn.LayerNorm(d_model)
        # Spec queries Spat (GQA)
        self.cross_attn_s2t = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_heads // 2)
        # Spat queries Spec (GQA)
        self.cross_attn_t2s = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_heads // 2)
        # Static Learnable Gates
        self.gate_spec = nn.Parameter(torch.zeros(1))
        self.gate_spat = nn.Parameter(torch.zeros(1))
    
    def forward(self, x_spec, x_spat):
        # x_spec, x_spat: (B, S, C, D)
        B, S, C, D = x_spec.shape
        
        # Reshape to (B*C, S, D) for per-channel cross-attention
        spec_flat = x_spec.permute(0, 2, 1, 3).contiguous().view(B*C, S, D)
        spat_flat = x_spat.permute(0, 2, 1, 3).contiguous().view(B*C, S, D)
        
        # Spectral stream queries Spatial stream
        spec_q = self.norm_spec_q(spec_flat)
        spat_kv = self.norm_spat_kv(spat_flat)
        delta_spec, _ = self.cross_attn_s2t(spec_q, spat_kv, spat_kv, need_weights=False)
        
        # Spatial stream queries Spectral stream
        spat_q = self.norm_spat_q(spat_flat)
        spec_kv = self.norm_spec_kv(spec_flat)
        delta_spat, _ = self.cross_attn_t2s(spat_q, spec_kv, spec_kv, need_weights=False)
        
        # Reshape back and gated residual
        delta_spec = delta_spec.view(B, C, S, D).permute(0, 2, 1, 3)
        delta_spat = delta_spat.view(B, C, S, D).permute(0, 2, 1, 3)
        
        return x_spec + self.gate_spec.sigmoid() * delta_spec, \
               x_spat + self.gate_spat.sigmoid() * delta_spat


class SpectralMambaClassifier(nn.Module):
    def __init__(self, num_classes=3, d_model=512, depth=8, dropout=0.0, drop_path_rate=0.0, use_checkpoint=False):
        super().__init__()
        
        # Gradient Checkpointing Flag
        self.use_checkpoint = use_checkpoint
        
        # Input: 110 bands + 9 VIs = 119 channels
        self.in_channels = 119
        self.d_model = d_model
        
        # 1. Patch Embedding (Spatial Reduction)
        self.patch_embed = PatchEmbed(in_channels=1, d_model=d_model)
        
        # 2. Physical Wavelength Embedding
        self.pos_encoder = GaussianFourierMapping(input_dim=1, mapping_size=d_model//2, scale=1.0)
        self.pos_proj = nn.Linear(d_model, d_model)
        
        # Wavelength Sensing (Channel Attention)
        self.wavelength_sensing = ImplicitWavelengthSensing(self.in_channels, WAVELENGTHS)
        
        # 3. Dual-Stream Backbone (Sparse Fusion)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        
        # Spectral Stream: Pure Mamba (scans along channels)
        self.spectral_layers = nn.ModuleList([
            SpectralMambaBlock(d_model=d_model, d_state=16, d_conv=4, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        # Spatial Stream: Pure Transformer (attends across spatial positions)
        self.spatial_layers = nn.ModuleList([
            SpatialTransformerBlock(d_model=d_model, num_heads=8, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        # Sparse Fusion: Bottleneck at layer 4 (mid-level), Cross-Attention at layer 8 (deep)
        self.fusion_indices = [3, 7]
        self.cross_fusions = nn.ModuleList([
            BottleneckFusionBlock(d_model=d_model),    # Layer 4: lightweight
            CrossFusionBlock(d_model=d_model, num_heads=8),  # Layer 8: rich
        ])
        
        self.norm_f = nn.LayerNorm(d_model)
        
        # Learnable per-dimension merge gate for dual-stream fusion
        # sigmoid(0)=0.5 → equal weighting initially
        self.merge_gate = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        
        # 4. Classification Head
        # Double Insurance: Max Pool (for spots) + Avg Pool (for general health)
        self.dropout = nn.Dropout(dropout)
        
        # Dynamic Spectral Fusion (SE-Block style content-aware weighting)
        self.spectral_fusion = DynamicSpectralFusion(d_model, num_bands=self.in_channels)
        
        self.head = nn.Linear(d_model * 2, num_classes)
        
        # Wavelength buffer for positional encoding
        w_tensor = torch.tensor(WAVELENGTHS, dtype=torch.float32)
        w_norm = (w_tensor - w_tensor.min()) / (w_tensor.max() - w_tensor.min() + 1e-6)
        self.register_buffer('w_norm', w_norm)
        
        # Spatial Position Embedding
        # Default size 64 corresponds to 8x8 patches (32x32 input)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 64, 1, d_model))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=.02)

    def forward_encoder(self, x, mask_ratio=0.0, spectral_mask_ratio=0.0):
        # x: (B, 116, H, W)
        B, C, H, W = x.shape
        
        # 1. Implicit Sensing
        x = self.wavelength_sensing(x)
        
        # 2. Patch Embedding (Spatial Reduction 4x)
        # Reshape to (B*C, 1, H, W) to process each band
        x = x.view(B * C, 1, H, W)
        x = self.patch_embed(x)
        
        # Get new spatial dims
        _, H_new, W_new, D = x.shape # D is d_model
        
        # 3. Reshape for S3M (Spatial-Spectral Sequential Mamba)
        # We need to maintain BOTH Spatial and Spectral dimensions separate.
        # (B*C, H', W', D) -> (B, C, H', W', D) -> (B, S, C, D)
        x = x.reshape(B, C, H_new, W_new, D)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H_new * W_new, C, D)
        
        S = x.shape[1]
        
        # 4. Add Positional Encoding
        # 4.1 Spatial Position Encoding (shared across all bands)
        # Handle interpolation if spatial sequence length S differs from parameter size
        if self.spatial_pos_embed.shape[1] != S:
            # Assume square patches for 2D interpolation
            orig_size = int(math.sqrt(self.spatial_pos_embed.shape[1]))
            new_size = int(math.sqrt(S))
            
            # (1, S_orig, 1, D) -> (1, D, H_orig, W_orig)
            pe = self.spatial_pos_embed.squeeze(2).permute(0, 2, 1)
            pe = pe.reshape(1, D, orig_size, orig_size)
            
            # Interpolate to new spatial dimensions
            pe = F.interpolate(pe, size=(new_size, new_size), mode='bicubic', align_corners=False)
            
            # (1, D, H_new, W_new) -> (1, S_new, 1, D)
            current_spatial_pos_embed = pe.reshape(1, D, S).permute(0, 2, 1).unsqueeze(2)
        else:
            current_spatial_pos_embed = self.spatial_pos_embed
            
        # Add spatial position encoding (broadcast across channel dimension)
        # (1, S, 1, D) -> (B, S, C, D)
        x = x + current_spatial_pos_embed
        
        # 4.2 Spectral (Wavelength) Position Encoding
        w_full = self.w_norm
        if w_full.size(0) != C:
             diff = C - w_full.size(0)
             w_ext = torch.ones(diff).to(x.device) * 2.0 
             w_full = torch.cat([w_full.to(x.device), w_ext], dim=0)
        else:
             w_full = w_full.to(x.device)

        pos_emb = self.pos_encoder(w_full) 
        pos_emb = self.pos_proj(pos_emb) # (C, d_model)
        
        # Add spectral encoding (broadcast across spatial dimension)
        # (1, 1, C, D) -> (B, S, C, D)
        x = x + pos_emb.unsqueeze(0).unsqueeze(0)

        # --- MAE MASKING ---
        mask, ids_restore, ids_keep = None, None, None
        if mask_ratio > 0:
            # Masking Spatial Dimension S
            # x: (B, S, C, D)
            # We want to keep (1-mask_ratio) * S patches per batch item
            
            len_keep = int(S * (1 - mask_ratio))
            
            # Generate random noise (B, S)
            noise = torch.rand(B, S, device=x.device)
            
            # Sort noise to get random indices
            ids_shuffle = torch.argsort(noise, dim=1) # (B, S)
            ids_restore = torch.argsort(ids_shuffle, dim=1) # (B, S)
            
            ids_keep = ids_shuffle[:, :len_keep] # (B, len_keep)
            
            # Gather kept patches
            # x is (B, S, C, D). We need to gather dim 1 using ids_keep
            # ids_keep extended: (B, len_keep, C, D)
            
            # Mask Token Mask (0=visible, 1=masked)
            mask_token_mask = torch.ones([B, S], device=x.device)
            mask_token_mask[:, :len_keep] = 0
            mask = torch.gather(mask_token_mask, dim=1, index=ids_restore)

            # OPTIMIZATION: Vectorized Gather (No Python Loop)
            ids_keep_expanded = ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, D)
            x = torch.gather(x, dim=1, index=ids_keep_expanded)
            
            # Update S
            S = x.shape[1]

        # --- SPECTRAL CONTINUOUS BAND MASK ---
        # Masks a contiguous segment of spectral bands by zeroing them out.
        # This forces the model to infer spectral shape from remaining bands.
        if spectral_mask_ratio > 0 and self.training:
            spectral_mask_len = max(1, int(C * spectral_mask_ratio))
            max_start = C - spectral_mask_len
            start_idx = random.randint(0, max_start)
            # Create spectral mask: (1, 1, C, 1) -> broadcast over (B, S, C, D)
            spec_mask = torch.ones(1, 1, C, 1, device=x.device, dtype=x.dtype)
            spec_mask[:, :, start_idx:start_idx + spectral_mask_len, :] = 0.0
            x = x * spec_mask  # Zero out masked spectral bands
        
        # 6. Dual-Stream Sparse Fusion Loop
        # Two independent streams with cross-attention fusion at layers 4 and 8
        x_spec = x  # Spectral stream (Mamba)
        x_spat = x  # Spatial stream (Transformer)
        
        fusion_idx = 0
        for i in range(len(self.spectral_layers)):
            # Run both streams independently
            if self.use_checkpoint and self.training:
                x_spec = checkpoint(self.spectral_layers[i], x_spec, use_reentrant=False)
                x_spat = checkpoint(self.spatial_layers[i], x_spat, use_reentrant=False)
            else:
                x_spec = self.spectral_layers[i](x_spec)
                x_spat = self.spatial_layers[i](x_spat)
            
            # Cross-Attention Fusion at designated layers
            if fusion_idx < len(self.fusion_indices) and i == self.fusion_indices[fusion_idx]:
                x_spec, x_spat = self.cross_fusions[fusion_idx](x_spec, x_spat)
                fusion_idx += 1
        
        # Learnable Merge: per-dimension gating
        # alpha ∈ (0,1) per feature dim, learned during training
        alpha = self.merge_gate.sigmoid()  # (1, 1, 1, D)
        x = alpha * x_spec + (1 - alpha) * x_spat
        
        x = self.norm_f(x)
        
        return x, mask, ids_restore, H_new, W_new

    def forward(self, x):
        # x: (B, 116, H, W)
        
        # 1. Encoder (No Masking)
        x, _, _, H_new, W_new = self.forward_encoder(x, mask_ratio=0.0)
        B, S, C, D = x.shape
        
        # 7. Aggregate Spectral Dimension (Dynamic Content-Aware Weighting)
        x_fused = self.spectral_fusion(x)  # (B, S, D)
        
        # Flatten S for pooling (If not already flat)
        x_fused = x_fused.contiguous().view(B * S, D) 
        
        # 8. Restore Spatial Context & Global Pooling
        # (B*S, D) -> (B, S, D) -> (B, D, H', W')
        x = x_fused.view(B, H_new, W_new, D).permute(0, 3, 1, 2)
        
        # 9. Classification with Double Pooling
        x_max = F.adaptive_max_pool2d(x, (1, 1)).flatten(1) 
        x_avg = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = torch.cat([x_max, x_avg], dim=1)
        
        logits = self.head(self.dropout(x))
        return logits

class S3M_MAE(nn.Module):
    """
    Masked Autoencoder for S3M.
    """
    def __init__(self, encoder, decoder_dim=256, decoder_depth=2, spectral_grad_weight=0.1):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model
        
        # Spectral Gradient Loss Weight
        self.spectral_grad_weight = spectral_grad_weight
        

        
        # Decoder (Lightweight Mamba)
        self.decoder_embed = nn.Linear(self.d_model, decoder_dim, bias=True)
        self.mask_token_dec = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # Decoder Position Encoding
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 64, decoder_dim), requires_grad=False) # Fixed 8x8 patches
        
        decoder_layers = []
        for _ in range(decoder_depth):
             # Simple Decoder: Alternating blocks or just Spatial blocks?
             # Reconstruction is spatial.
             decoder_layers.append(SpatialMambaBlock(d_model=decoder_dim, d_state=16))
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        
        # Prediction Head: output per-patch pixel reconstruction
        # Output dim = patch_size^2 * num_channels = 16 * 119 = 1904
        patch_size = 4
        in_channels = 119
        self.pred_head = nn.Linear(decoder_dim, patch_size**2 * in_channels, bias=True)
        
        # UPGRADE: Bottleneck Adapter with Low-Rank Decomp
        # Replaces simple mean() with Linear projection
        # Input: C * D (Encoder Feature per spectral band) -> Output: Decoder Dim
        # 60928 -> 256 is too big (15M params) -> 60928 -> 64 -> 256
        reduction_dim = 64
        self.bottleneck_adapter = nn.Sequential(
            nn.Linear(in_channels * self.d_model, reduction_dim),
            nn.GELU(),
            nn.Linear(reduction_dim, decoder_dim)
        )
        
        # Intitialize Pos Embed (Simple Sinusoid for patches 0-63)
        self.initialize_decoder_pos_embed()

    def initialize_decoder_pos_embed(self):
        # ... logic to init 1D sine-cosine encoding for 64 positions ...
        # Simplified:
        pos = torch.arange(64).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.decoder_embed.out_features, 2).float() * -(math.log(10000.0) / self.decoder_embed.out_features))
        pe = torch.zeros(64, self.decoder_embed.out_features)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.decoder_pos_embed.data.copy_(pe.unsqueeze(0))

    def forward(self, x, mask_ratio=0.75, spectral_mask_ratio=0.0):
        # x: (B, 119, 32, 32)
        
        # 1. Encoder Forward (Masked)
        # latent: (B, len_keep, C, D)
        latent, mask, ids_restore, _, _ = self.encoder.forward_encoder(
            x, mask_ratio=mask_ratio, spectral_mask_ratio=spectral_mask_ratio
        )
        
        B, S_keep, C, D = latent.shape
        
        # 2. Spectral Aggregation for Latent (Information Bottleneck Fix)
        # Old: latent_spatial = latent.mean(dim=2)
        # New: Flatten Spectral & Feature dims -> Linear Projection
        
        # (B, S_keep, C, D) -> (B, S_keep, C*D)
        latent_flat = latent.reshape(B, S_keep, C * D)
        
        # Project to Decoder Dim (Preserves Spectral Info)
        # (B, S_keep, D_dec)
        # Note: We skip self.decoder_embed since adapter does the projection
        x_dec = self.bottleneck_adapter(latent_flat) 
        
        # 4. Append Mask Tokens
        mask_tokens = self.mask_token_dec.repeat(B, ids_restore.shape[1] - S_keep, 1) # (B, S_mask, D_dec)
        
        # Concat (B, S_mask+S_keep, D_dec) (This is effectively shuffled order)
        # We need to restore Original Order
        
        # Actually easier: Create Full Tensor
        x_full = torch.cat([x_dec, mask_tokens], dim=1) # (B, S, D_dec) BUT SHUFFLED
        
        # Unshuffle
        # We need to reorder according to ids_restore
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2]))
        
        # 5. Add Pos Embed
        x_full = x_full + self.decoder_pos_embed
        
        # 6. Decoder Layers
        # These are SpatialMambaBlocks, expecting (B, S, C, D). 
        # But we aggregated C. So input is (B, S, D_dec).
        # We need to adapt SpatialMambaBlock or just use it with C=1.
        x_dec_run = x_full.unsqueeze(2) # (B, S, 1, D_dec)
        for layer in self.decoder_layers:
            x_dec_run = layer(x_dec_run)
            
        x_rec = self.decoder_norm(x_dec_run.squeeze(2)) # (B, S, D_dec)
        
        # 7. Predict
        # Output: (B, S, 119 * 16)
        pred = self.pred_head(x_rec)
        
        # 8. Loss
        loss = self.forward_loss(x, pred, mask)
        
        return loss, pred, mask

    def patchify(self, imgs):
        """
        imgs: (N, 119, H, W)
        x: (N, L, patch_size**2 * 119)
        """
        p = 4
        B, C, H, W = imgs.shape # (B, 119, 32, 32)
        h = w = H // p
        x = imgs.reshape(shape=(B, C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcqp', x)
        x = x.reshape(shape=(B, h * w, C * p ** 2))
        return x

    def spectral_grad_loss(self, pred, target):
        """
        Spectral Gradient Loss: first-order spectral derivative MSE.
        Forces the model to learn spectral shape, not just pixel values.
        
        pred, target: (B, S, C * p^2) where C=119, p=4
        Returns: (B, S) loss per patch
        """
        p2 = 16  # patch_size^2 = 4*4
        C = pred.shape[-1] // p2  # 119
        
        # Reshape to separate channels: (B, S, C, p^2)
        pred_r = pred.view(pred.shape[0], pred.shape[1], C, p2)
        target_r = target.view(target.shape[0], target.shape[1], C, p2)
        
        # First-order spectral derivative along channel dim
        grad_pred = pred_r[:, :, 1:, :] - pred_r[:, :, :-1, :]    # (B, S, C-1, p^2)
        grad_target = target_r[:, :, 1:, :] - target_r[:, :, :-1, :]
        
        # MSE of gradients, mean over channel and pixel dims -> (B, S)
        grad_loss = (grad_pred - grad_target) ** 2
        grad_loss = grad_loss.mean(dim=-1).mean(dim=-1)  # (B, S)
        return grad_loss

    def forward_loss(self, imgs, pred, mask):
        # imgs: (B, 119, 32, 32)
        # pred: (B, 64, 119*16)
        # mask: (B, 64) 0=Keep, 1=Masked
        
        target = self.patchify(imgs)
        
        # --- MAE PATCH NORMALIZATION ---
        # Normalize target per patch
        # mean: (B, L, 1), var: (B, L, 1)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        
        # Compute MSE: (B, S)
        mse = (pred - target) ** 2
        mse = mse.mean(dim=-1)  # (B, S) - mean over pixel/channel dims
        
        # Spectral Gradient Loss: (B, S)
        spec_grad = self.spectral_grad_loss(pred, target)
        
        # Combined Loss (per-patch)
        loss = mse + self.spectral_grad_weight * spec_grad
        
        # Masked Loss
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
