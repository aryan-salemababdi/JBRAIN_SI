import torch
from torch import nn

def create_patches(x: torch.Tensor, patch_size=32):
    # Add batch dimension if needed
    if x.dim() == 2:
        x = x.unsqueeze(0)
        removed_batch_dim = True
    else:
        removed_batch_dim = False

    if x.shape[2] % patch_size != 0:
        raise RuntimeError(
            f"Input length {x.shape[2]} is not divisible by patch size {patch_size}"
        )

    # x = [batch, channels, seq_len]

    # [batch, channels, num_patches, patch_size]
    patches = x.unfold(2, patch_size, patch_size)
    # [batch, num_patches, channels, patch_size]
    patches = patches.permute(0, 2, 1, 3)

    if removed_batch_dim:
        patches = patches.squeeze(0)
    return patches



# Fixed PositionalEncoder
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, seq_len, n=10000):
        super().__init__()
        self.embedding_dim = torch.tensor(embedding_dim)
        self.seq_len = torch.tensor(seq_len)
        self.n = torch.tensor(n)

        # Precompute encoding matrix during initialization
        encoding_matrix = torch.zeros(seq_len, embedding_dim)
        for pos in range(seq_len):
            for i in range(embedding_dim // 2):
                denominator = torch.pow(n, torch.tensor(2 * i) / embedding_dim)
                encoding_matrix[pos, 2 * i] = torch.sin(pos / denominator)
                encoding_matrix[pos, 2 * i + 1] = torch.cos(pos / denominator)

        # Register as buffer to move with device
        self.register_buffer(
            "encoding_matrix", encoding_matrix.unsqueeze(0)
        )  # [1, seq_len, dim]

    def forward(self, x):
        # x shape: [batch, seq_len, dim] or [seq_len, dim]
        if x.dim() == 2:
            return x + self.encoding_matrix.squeeze(0)
        else:
            return x + self.encoding_matrix


if __name__ == "__main__":
    embedding_dim = 4
    pe = PositionalEncoder(embedding_dim, 10)
    # Test
    print(pe(torch.randn((10, embedding_dim)), 0))


class ClassToken(nn.Module):
    """Add learnable class token for classification"""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, num_patches, D)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls_tokens, x], dim=1)


class EmbeddingLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        patch_size=32,
        positional_encoding=False,
        with_class_token=False,
        number_of_channels=2,
        embedding_dim=32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.positional_encoding = positional_encoding
        self.with_class_token = with_class_token

        if self.positional_encoding:
            self.pe = PositionalEncoder(embedding_dim, self.num_patches)

        self.flatten = nn.Flatten(start_dim=1)
        self.cls_token = ClassToken(embed_dim=embedding_dim)

        self.encoder = nn.Linear(patch_size, embedding_dim // number_of_channels)

    def forward(self, x: torch.Tensor):
        ## What do?
        # 1. Create patches of input signals
        # 2. Combine patches and batch dim, to perform embedding for each patch separately
        # 3. Combine channels and perform embedding
        # 4. reshape back to [batch, num_patches, embedding_dim]. Transformer treat the the embedding as a sequence
        # 5. If positional encoding is active, add positional encoding
        # 6. If class token is active, add class token
        # 7. Return the output

        original_shape = x.shape
        # Add batch dim if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Create patches: [batch, num_patches, channels, patch_size]
        patches = create_patches(x, self.patch_size)
        batch_size, num_patches, channels, _ = patches.shape

        # Combine batch and num_patches
        x = patches.reshape(batch_size * num_patches, channels, self.patch_size)

        x = self.encoder(x)

        x = x.view(batch_size, num_patches, -1)

        if self.positional_encoding:
            x = self.pe(x)

        # Remove batch dim if input didn't have it
        if len(original_shape) == 2:
            x = x.squeeze(0)

        # Add class token
        if self.with_class_token:
            x = self.cls_token(x)  # [batch, num_patches*2, embedding_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        with_class_token=True,
        feedforward_dim=256,
        num_heads=8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.with_class_token = with_class_token

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.LeakyReLU(),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]

        x = self.layer_norm1(x)

        x_feedforward = self.feedforward(x)
        x = x + x_feedforward

        x = self.layer_norm2(x)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        patch_size=32,
        embedding_dim=64,
        positional_encoding=True,
        with_class_token=True,
        feedforward_dim=256,
        num_heads=8,
        num_encoders=12,
        number_of_channels=2,
        num_classes=10,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.with_class_token = with_class_token

        self.normalize = nn.LayerNorm(input_dim)

        self.embedding = EmbeddingLinear(
            positional_encoding=positional_encoding,
            embedding_dim=embedding_dim,
            input_dim=input_dim,
            patch_size=patch_size,
            with_class_token=with_class_token,
            number_of_channels=number_of_channels,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.num_encoders = num_encoders
        for i in range(num_encoders):
            setattr(
                self,
                f"encoder{i}",
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    with_class_token=with_class_token,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                ),
            )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, num_classes),
        )

    def encode(self, x):
        x = self.embedding(x)
        for i in range(self.num_encoders):
            x = getattr(self, f"encoder{i}")(x)
            x = self.dropout(x)

        if self.with_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return x

    def forward(self, x):
        x = torch.fft.fft(x, dim=-1)
        x = torch.abs(x)

        x = self.normalize(x)
        x = self.embedding(x)
        for i in range(self.num_encoders):
            x = getattr(self, f"encoder{i}")(x)
            x = self.dropout(x)

        if self.with_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        x = self.classifier(x)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model = TransformerEncoderClassifier(
        input_dim=2048,
        patch_size=16,
        embedding_dim=64,
        positional_encoding=False,
        num_classes=10,
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(32, 2, 2048)
    y = model(x)
    print(y.shape)
