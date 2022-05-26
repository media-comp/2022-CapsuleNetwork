import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def normalize():
    operations = tf.keras.layers.Rescaling(1.0 / 255)

    def img_norm(x: tf.Tensor):
        return operations(x)

    return img_norm




def find_visuals(model: tf.keras.Model, digit, dim, lb, ub, intervals):
    img_norm = normalize()
    visuals = []
    _, test_set = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True)

    for x, y in test_set.shuffle(1024).batch(1):
        if y.numpy()[0] == digit:
            class_vector = (model(img_norm(x))[0]).numpy()[0]
            if np.argmax(np.linalg.norm(class_vector, axis=-1)) == digit:
                #   visuals.append(x.numpy()[0][..., 0])

                fig, ax = plt.subplots(1, figsize=(9, 9))
                ax.imshow(x.numpy()[0][..., 0], cmap="gray")
                ax.axis("off")
                file_name = f"static/img0.png"
                plt.savefig(file_name, bbox_inches="tight")

                original_value = class_vector[digit, dim]
                lower = lb
                upper = ub
                interval = (upper - lower) / intervals
                for i in range(intervals+1):
                    tweaked_vector = class_vector.copy()
                    tweaked_vector[digit, dim] = original_value + lower + intervals * i
                    img = model.reconstruct(np.expand_dims(tweaked_vector, 0)).numpy()[0][..., 0]

                    fig, ax = plt.subplots(1, figsize=(9, 9))
                    ax.imshow(img, cmap="gray")
                    ax.axis("off")
                    index = i + 1
                    file_name = f"static/img{index}.png"
                    plt.savefig(file_name, bbox_inches="tight")
                break
    return original_value + lower, original_value + upper, interval


# def pred_save_image(model, digit, dim, lb, ub):
#     model = tf.keras.models.load_model("../model")
#     _, test_set = tfds.load(
#         "mnist", split=["train", "test"], shuffle_files=True, as_supervised=True
#     )

#     visuals, lower, upper = find_visuals(test_set, model, digit, dim, lb, ub)

#     for index, v in enumerate(visuals):
#         fig, ax = plt.subplots(1, figsize=(3, 3))
#         ax.imshow(v, cmap="gray")
#         ax.axis("off")
#         file_name = f"static/img{index}.png"
#         plt.savefig(file_name, bbox_inches="tight")

#     return lower, upper
