from tf_video_models.builders import get_eanet, get_mlp_mixer
def test_eanet():
    input_shape = (128, 128, 128, 1)
    patch_size = (16, 16, 16)
    num_classes = 2

    model = get_eanet(input_shape, patch_size, num_classes)
    print(model.summary())

def test_mlp_mixer():
    input_shape = (128, 128, 128, 1)
    patch_size = (16, 16, 16)
    num_classes = 2

    model = get_mlp_mixer(input_shape, patch_size, num_classes)
    print(model.summary())

if __name__ == '__main__':
    # test_eanet()
    test_mlp_mixer()