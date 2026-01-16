def convert_sam_to_class_decoder(sam, num_classes=7):
    md = sam.mask_decoder
    transformer_dim = md.transformer_dim

    md.num_mask_tokens = num_classes
    md.mask_tokens = nn.Embedding(num_classes, transformer_dim)

    md.output_hypernetworks_mlps = nn.ModuleList([
        nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim // 8),
        )
        for _ in range(num_classes)
    ])

    return sam
