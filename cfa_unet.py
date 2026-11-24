import torch
from torch import nn


class CrossFrequencyAttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            nn.ModuleList([
                self._conv_block(1, 64),
                self._conv_block(64, 128),
                self._conv_block(128, 256),
                self._conv_block(256, 512),
            ])
            for _ in range(3)
        ])
        self.pool = nn.MaxPool2d(2)

        self.frequency_projections = nn.ModuleList([
            nn.Conv2d(512, 96, kernel_size=1)
            for _ in range(3)
        ])

        self.attention_norm = nn.LayerNorm(96)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=96,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(96)
        self.ffn = nn.Sequential(
            nn.Linear(96, 384),
            nn.GELU(),
            nn.Linear(384, 96),
        )

        self.decoder_blocks = nn.ModuleList([
            self._conv_block(288, 512),
            self._conv_block(1024, 256),
            self._conv_block(512, 128),
            self._conv_block(256, 64),
        ])
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        ])
        self.output_head = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        encoder_features = []
        for frequency, encoder in enumerate(self.encoders):
            features = x[:, frequency:frequency + 1]
            branch_features = []
            for level, block in enumerate(encoder):
                if level > 0:
                    features = self.pool(features)
                features = block(features)
                branch_features.append(features)
            encoder_features.append(branch_features)

        tokens = [
            projection(features[-1]).flatten(2).transpose(1, 2)
            for projection, features in zip(
                self.frequency_projections, encoder_features
            )
        ]
        normalized_tokens = [self.attention_norm(t) for t in tokens]
        height, width = encoder_features[0][-1].shape[-2:]

        attended_maps = []
        for frequency in range(3):
            context = torch.stack([
                normalized_tokens[other]
                for other in range(3)
                if other != frequency
            ], dim=2).reshape(-1, 2, 96)

            attended, _ = self.cross_attention(
                query=normalized_tokens[frequency].reshape(-1, 1, 96),
                key=context,
                value=context,
                need_weights=False,
            )
            enriched = tokens[frequency] + attended.reshape_as(tokens[frequency])
            enriched = enriched + self.ffn(self.ffn_norm(enriched))
            attended_maps.append(
                enriched.transpose(1, 2).reshape(x.shape[0], 96, height, width)
            )

        features = self.decoder_blocks[0](torch.cat(attended_maps, dim=1))
        for upsampler, decoder, level in zip(
            self.upsamplers, self.decoder_blocks[1:], (2, 1, 0)
        ):
            features = upsampler(features)
            skip_features = [branch[level] for branch in encoder_features]
            features = decoder(torch.cat([features, *skip_features], dim=1))

        return torch.sigmoid(self.output_head(features))
