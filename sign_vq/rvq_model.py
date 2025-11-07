"""RVQ-VAE (Residual Vector Quantized Variational Autoencoder) for motion data."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional


class VectorQuantizer(nn.Module):
    """Vector Quantization layer.

    Args:
        num_embeddings: Size of the codebook
        embedding_dim: Dimension of embeddings
        commitment_cost: Weight for commitment loss
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        """Quantize input.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            quantized: Quantized tensor
            indices: Codebook indices
            commitment_loss: Commitment loss
        """
        # Flatten to (batch * seq_len, embedding_dim)
        batch_size, seq_len, _ = x.shape
        flat_x = x.view(-1, self.embedding_dim)

        # Calculate distances to embeddings
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )

        # Get nearest embedding indices
        indices = torch.argmin(distances, dim=1)
        quantized_flat = self.embeddings(indices)

        # Reshape back
        quantized = quantized_flat.view(batch_size, seq_len, self.embedding_dim)

        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, indices.view(batch_size, seq_len), commitment_loss


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple quantization layers.

    Args:
        num_quantizers: Number of residual quantization layers
        num_embeddings: Size of each codebook
        embedding_dim: Dimension of embeddings
        commitment_cost: Weight for commitment loss
    """

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create multiple quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """Apply residual quantization.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim)

        Returns:
            quantized: Final quantized tensor
            all_indices: List of indices from each quantizer
            total_commitment_loss: Sum of commitment losses
        """
        quantized = torch.zeros_like(x)
        residual = x
        all_indices = []
        total_commitment_loss = 0

        for quantizer in self.quantizers:
            q, indices, loss = quantizer(residual)
            quantized = quantized + q
            residual = residual - q
            all_indices.append(indices)
            total_commitment_loss = total_commitment_loss + loss

        return quantized, all_indices, total_commitment_loss

    def encode(self, x):
        """Encode input to codebook indices."""
        _, all_indices, _ = self.forward(x)
        # Stack indices: (batch, seq_len, num_quantizers)
        return torch.stack(all_indices, dim=-1)

    def decode(self, indices):
        """Decode codebook indices to embeddings.

        Args:
            indices: Tensor of shape (batch, seq_len, num_quantizers)

        Returns:
            Reconstructed embeddings
        """
        batch_size, seq_len, num_q = indices.shape
        assert num_q == self.num_quantizers

        quantized = torch.zeros(
            batch_size, seq_len, self.embedding_dim,
            device=indices.device, dtype=self.quantizers[0].embeddings.weight.dtype
        )

        for i, quantizer in enumerate(self.quantizers):
            codes = quantizer.embeddings(indices[:, :, i])
            quantized = quantized + codes

        return quantized


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class RVQ_VAE(nn.Module):
    """RVQ-VAE model for motion data.

    Args:
        input_dim: Input dimension (num_keypoints * 3)
        hidden_dim: Hidden dimension
        num_quantizers: Number of residual quantizers
        codebook_size: Size of each codebook
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int = 75,  # 25 keypoints * 3
        hidden_dim: int = 512,
        num_quantizers: int = 4,
        codebook_size: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # Store all arguments for logging
        self.args_dict = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_quantizers': num_quantizers,
            'codebook_size': codebook_size,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout
        }

        # Encoder
        self.encoder_input = nn.Linear(input_dim, hidden_dim)
        self.encoder_pos = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Residual Vector Quantizer
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=codebook_size,
            embedding_dim=hidden_dim,
            commitment_cost=0.25
        )

        # Decoder
        self.decoder_pos = PositionalEncoding(hidden_dim, dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Encode input to latent codes.

        Args:
            x: Input tensor of shape (batch, seq_len, num_keypoints, 3)

        Returns:
            Codebook indices of shape (batch, seq_len, num_quantizers)
        """
        batch_size, seq_len, num_kp, dims = x.shape
        assert num_kp * dims == self.input_dim

        # Flatten keypoints
        x_flat = x.view(batch_size, seq_len, -1)

        # Encode
        h = self.encoder_input(x_flat)
        h = self.encoder_pos(h)
        h = self.encoder(h)

        # Quantize
        indices = self.rvq.encode(h)

        return indices

    def decode(self, indices):
        """Decode codebook indices to motion.

        Args:
            indices: Codebook indices of shape (batch, seq_len, num_quantizers)

        Returns:
            Reconstructed motion of shape (batch, seq_len, num_keypoints, 3)
        """
        # Decode from codebook
        h = self.rvq.decode(indices)

        # Decode to output
        h = self.decoder_pos(h)
        h = self.decoder(h)
        x_recon = self.decoder_output(h)

        # Reshape to (batch, seq_len, num_keypoints, 3)
        batch_size, seq_len, _ = x_recon.shape
        num_keypoints = self.input_dim // 3
        x_recon = x_recon.view(batch_size, seq_len, num_keypoints, 3)

        return x_recon

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, num_keypoints, 3)

        Returns:
            x_recon: Reconstructed tensor
            commitment_loss: Commitment loss from quantizer
        """
        batch_size, seq_len, num_kp, dims = x.shape

        # Flatten keypoints
        x_flat = x.view(batch_size, seq_len, -1)

        # Encode
        h = self.encoder_input(x_flat)
        h = self.encoder_pos(h)
        h = self.encoder(h)

        # Quantize
        h_quantized, _, commitment_loss = self.rvq(h)

        # Decode
        h_dec = self.decoder_pos(h_quantized)
        h_dec = self.decoder(h_dec)
        x_recon = self.decoder_output(h_dec)

        # Reshape
        x_recon = x_recon.view(batch_size, seq_len, num_kp, dims)

        return x_recon, commitment_loss


class RVQ_VAE_Lightning(pl.LightningModule):
    """PyTorch Lightning wrapper for RVQ-VAE."""

    def __init__(
        self,
        model: RVQ_VAE,
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        reconstruction_weight: float = 1.0,
        commitment_weight: float = 0.25
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.reconstruction_weight = reconstruction_weight
        self.commitment_weight = commitment_weight

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x_recon, commitment_loss = self(batch)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, batch)

        # Total loss
        loss = self.reconstruction_weight * recon_loss + self.commitment_weight * commitment_loss

        # Logging
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/commitment_loss', commitment_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x_recon, commitment_loss = self(batch)

        recon_loss = F.mse_loss(x_recon, batch)
        loss = self.reconstruction_weight * recon_loss + self.commitment_weight * commitment_loss

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/commitment_loss', commitment_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * step / 100000)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
