
class SAMFormationModel(nn.Module):
    def __init__(self, sam, n_classes=20, k_decoder_layers=3):
        super().__init__()
        self.sam = sam

        self.decoder = SAMDecoderForClassification(
            sam=sam,
            k_layers=k_decoder_layers
        )
        self.geo = GeometryEnhancer(d_model=256)
        self.head = FormationHead(
            d_model=256,
            n_classes=n_classes
        )

    def forward(self, image, points_xy, points_label, valid_mask=None):
        image_embeddings = self.sam.image_encoder(image)

        image_pe = self.sam.prompt_encoder.get_dense_pe()

        sparse_pe, _ = self.sam.prompt_encoder(
            points=(points_xy, points_label),
            boxes=None,
            masks=None
        )

        N = points_xy.shape[1]
        tokens = sparse_pe[:, :N, :] 

        tokens = self.decoder(
            tokens,
            image_embeddings.detach(),
            image_pe
        )

        tokens = self.geo(tokens, points_xy, image.shape[-2:])

        return self.head(tokens, valid_mask)
