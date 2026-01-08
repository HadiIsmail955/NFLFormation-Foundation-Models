import torch.nn as nn
import torch.nn.functional as F

class SAMClassificationDecoder(nn.Module):
    def __init__(self, sam, k_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList(
            sam.mask_decoder.transformer.layers[:k_layers]
        )

    def forward(self, prompt_tokens, image_embeddings, image_pe):
        tokens = prompt_tokens
        for blk in self.blocks:
            tokens, _ = blk(tokens, image_embeddings, image_pe)
        return tokens
