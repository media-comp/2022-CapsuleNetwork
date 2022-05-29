import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from typing import Union

parser = argparse.ArgumentParser(description='Visualization of recognizing overlayed images in MNIST')
parser.add_argument('--model-path', type=str, default='./model', help='path to load the trained model, ./model by default')
args = parser.parse_args()


def generate_overlay(images: Union[np.ndarray, tf.Tensor]):
    overlay = np.sum(images, axis=0, keepdims=True)
    overlay = (1 - (overlay > 255)) * overlay + (overlay > 255) * 255
    return overlay


def normalize():
    operations = tf.keras.layers.Rescaling(1./255)

    def img_norm(x: tf.Tensor):
        return operations(x)

    return img_norm


def reconstruct_overlay(model: tf.keras.Model, overlay: Union[np.ndarray, tf.Tensor]):
    img_norm = normalize()

    class_vector, _ = model(img_norm(overlay))
    class_vector = class_vector.numpy()[0]
    one_class = np.argmax(np.linalg.norm(class_vector, axis=-1))
    another_class = class_vector.copy()
    another_class[one_class, :] = 0
    two_class = np.argmax(np.linalg.norm(another_class, axis=-1))

    recons_image = model.reconstruct(np.expand_dims(class_vector, axis=0)).numpy()[0]
    another_recons = model.reconstruct(np.expand_dims(another_class, axis=0)).numpy()[0]

    return recons_image, another_recons, one_class, two_class


if __name__ == '__main__':
    model = tf.keras.models.load_model(args.model_path)
    _, test_set = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

    for x, y in test_set.shuffle(1024).batch(2):
        y_one, y_two = y.numpy()
        if y_one == y_two:
            continue

        overlay = generate_overlay(x)

        recons_image, another_recons, one_class, two_class = reconstruct_overlay(model, overlay)

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(overlay[0, ..., 0], cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'Overlayed {y_one} {y_two}')
        axes[1].imshow(recons_image[..., 0], cmap='gray')
        axes[1].set_title(f'Reconstructed {one_class}')
        axes[1].axis('off')
        axes[2].imshow(another_recons[..., 0], cmap='gray')
        axes[2].set_title(f'Reconstructed {two_class}')
        axes[2].axis('off')
        fig.show()
        break
