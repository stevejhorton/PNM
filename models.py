import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.l1 = nn.Linear(d_model, dim_feedforward)
        self.l2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)
        return self.norm2(x + self.l2(torch.relu(self.l1(x))))

class Transformer(nn.Module):
    def __init__(self, layers=6, d_model=512, nhead=8, vocab=1000):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, nhead) for _ in range(layers)])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.head(self.blocks(self.emb(x)))
