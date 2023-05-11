from tensorflow import keras
from functools import partial
from tensorflow.keras import layers
from tf_video_models.functional.eanet import ea_encoder
from tf_video_models.layers.mlp_mixer import MLPMixerLayer
from tf_video_models.layers.patch_extractor import Patch3D, PatchEmbedding

def get_model(input_shape, 
              patch_size, 
              num_classes,
              encoder,
              num_blocks=16,
              **kwargs):
    inputs = layers.Input(shape=input_shape)
    # Image augment
    # Extract patches.
    x = Patch3D(patch_size)(inputs)
    # Create patch embedding.
    num_patches, embedding_dim = x.shape[1], x.shape[-1]
    x = PatchEmbedding(num_patches, embedding_dim)(x)
    # Create Transformer block.
    for _ in range(num_blocks):
        x = encoder(embedding_dim=embedding_dim, num_patches=num_patches)(x)

    x = layers.GlobalAvgPool1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_eanet(input_shape, 
              patch_size, 
              num_classes,
              num_blocks=5, 
              mlp_dim=512,
                num_heads=16,
                dim_coefficient=0.5,
                attention_dropout=0.25,
                projection_dropout=0.25,):
    encoder = partial(ea_encoder, 
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dim_coefficient=dim_coefficient,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout)
    return get_model(input_shape, patch_size, num_classes, encoder,
                     num_blocks=num_blocks, 
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dim_coefficient=dim_coefficient,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout)

def get_mlp_mixer(input_shape, 
              patch_size, 
              num_classes,
              num_blocks=5, 
              mlp_dim=512,
                dim_coefficient=0.5,
                dropout=0.25,
                ):
    encoder = partial(MLPMixerLayer, dropout_rate=dropout)
    return get_model(input_shape, patch_size, num_classes, encoder,
                     num_blocks=num_blocks, 
                    mlp_dim=mlp_dim,
                    dim_coefficient=dim_coefficient)
    

if __name__ == '__main__':
    input_shape = (1, 128, 128, 128, 1)
    patch_size = (16, 16, 16)
    num_classes = 2

    model = get_eanet(input_shape, patch_size, num_classes, ea_encoder)
    print(model.summary())