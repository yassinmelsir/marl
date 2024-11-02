from torch import nn


class TransformerSeq2Seq(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerSeq2Seq, self).__init__()
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, src, tgt):
        output = self.transformer(src, tgt)
        return self.fc_out(output)

    def get_embed_dim(self):
        return self.embed_dim