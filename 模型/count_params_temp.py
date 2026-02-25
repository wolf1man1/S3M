
import torch
from model import SpectralMambaClassifier, S3M_MAE

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    # Config
    d_model = 512
    depth = 8
    decoder_dim = 256
    decoder_depth = 4
    
    print(f"--- Configuration ---")
    print(f"d_model: {d_model}")
    print(f"depth: {depth}")
    print(f"decoder_dim: {decoder_dim}")
    print(f"decoder_depth: {decoder_depth}")
    print("-" * 20)

    # 1. Encoder (Classifier / Backbone)
    encoder = SpectralMambaClassifier(num_classes=3, d_model=d_model, depth=depth)
    e_total, e_trainable = count_parameters(encoder)
    print(f"Encoder (SpectralMambaClassifier) Params:")
    print(f"  Total: {e_total / 1e6:.2f} M")
    print(f"  Trainable: {e_trainable / 1e6:.2f} M")
    
    # 2. MAE (Pretrain Model)
    mae = S3M_MAE(encoder=encoder, decoder_dim=decoder_dim, decoder_depth=decoder_depth)
    m_total, m_trainable = count_parameters(mae)
    print(f"MAE (Pretrain Model) Params:")
    print(f"  Total: {m_total / 1e6:.2f} M")
    print(f"  Trainable: {m_trainable / 1e6:.2f} M")

if __name__ == "__main__":
    main()
