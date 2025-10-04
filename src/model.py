
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class GPTConfig:
    """Configuration for GPT model"""
    vocab_size = 32000
    block_size = 512
    n_embd = 512
    n_layer = 6
    n_head = 8
    dropout = 0.1

class GPT(nn.Module):
    """Decoder-only Transformer language model GPT-style"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb   = nn.Embedding(config.block_size, config.n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            activation="gelu",
            batch_first=True,
            dropout=config.dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx, targets=None):
        """Forward pass through GPT model"""
        B, T = idx.shape
        assert T <= self.config.block_size, "Sequence too long!"
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
