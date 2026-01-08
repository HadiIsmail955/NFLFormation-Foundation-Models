import torch.nn as nn

class SAMClassificationDecoder(nn.Module):
    def __init__(self, sam, k_layers: int = 3, trainable: bool = True):
        super().__init__()

        self.sam = sam
        self.k = k_layers

        self.blocks = nn.ModuleList(
            sam.mask_decoder.transformer.layers[:k_layers]
        )

        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool):
        for p in self.blocks.parameters():
            p.requires_grad = trainable

    def freeze(self):
        self.set_trainable(False)

    def unfreeze(self):
        self.set_trainable(True)

    def forward(self, prompt_tokens, image_embeddings, image_pe):
        tokens = prompt_tokens
        for blk in self.blocks:
            tokens, _ = blk(tokens, image_embeddings, image_pe)
        return tokens
