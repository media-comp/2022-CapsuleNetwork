import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualization of reconstructed images from vector outputs with tweaked value')
parser.add_argument('--model-path', type=str, default='./model', help='path to load the trained model, ./model by default')
parser.add_argument('--digit', type=int, default=6, help='the target digit to visualize on, 6 by default')
parser.add_argument('--dimension', type=int, default=5, help='the target dimension of the digit to visualize on, 5 by default')
parser.add_argument('--lower-difference', type=float, default=-0.25, help='the lower difference bound for the target value, -0.25 by default')
parser.add_argument('--upper-difference', type=float, default=0.25, help='the upper difference bound for the target value, 0.25 by default')
args = parser.parse_args()

digit = args.digit
dim = args.dimension

if digit < 0 or digit > 9:
    raise ValueError('The value of digit must be in [0, 9]')

if dim < 0 or dim > 15:
    raise ValueError('The value of dimension must be in [0, 15]')


def normalize():
    operations = tf.keras.layers.Rescaling(1./255)

    def img_norm(x: tf.Tensor):
        return operations(x)

    return img_norm


def find_visuals(dataset: tf.data.Dataset, model: tf.keras.Model):
    img_norm = normalize()
    visuals = []
    for x, y in dataset.shuffle(1024).batch(1):
        if y.numpy()[0] == digit:
            class_vector = (model(img_norm(x))[0]).numpy()[0]
            if np.argmax(np.linalg.norm(class_vector, axis=-1)) == digit:
                visuals.append(x.numpy()[0][..., 0])
                original_value = class_vector[digit, dim]
                lower = args.lower_difference
                upper = args.upper_difference
                intervals = (upper - lower) / 10
                for i in range(11):
                    tweaked_vector = class_vector.copy()
                    tweaked_vector[digit, dim] = original_value+lower+intervals*i
                    visuals.append(model.reconstruct(np.expand_dims(tweaked_vector, 0)).numpy()[0][..., 0])
                break

    return visuals, original_value+lower, original_value+upper


if __name__ == '__main__':
    model = tf.keras.models.load_model(args.model_path)
    _, test_set = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

    visuals, lower, upper = find_visuals(test_set, model)

    fig, axes = plt.subplots(1, 12, figsize=(12, 4))
    for v, ax in zip(visuals, axes):
        ax.imshow(v, cmap='gray')
        ax.axis('off')
    axes[0].set_title('Original')
    axes[1].set_title(f'{lower:.2f}')
    axes[-1].set_title(f'{upper:.2f}')
    fig.show()