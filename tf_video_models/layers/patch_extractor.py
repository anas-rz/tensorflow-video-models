import tensorflow as tf
from tensorflow.keras import layers


class Patch3D(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        assert len(patch_size) == 3, "Patch size must be of shape (d, h, w)"
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.extract_volume_patches(
            images,
            ksizes=(1, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1),
            strides=(1, self.patch_size[0], self.patch_size[1], self.patch_size[2], 1),
            # rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1] * patches.shape[2] * patches.shape[3]

        return tf.reshape(patches, (batch_size, patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)



if __name__ == '__main__':
    patches_layer = Patch3D((16, 16, 8))
    a = tf.random.uniform((1, 128, 128, 128, 1))
    out = patches_layer(a)
    num_patches = out.shape[1]
    patches_dim = out.shape[-1]
    out = PatchEmbedding(num_patches, patches_dim)(out)

    print(out.shape)