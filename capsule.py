import numpy as np
import tensorflow as tf
from typing import Union, List, Tuple, Optional


def shape_list(tensor: Union[tf.Tensor, tf.Variable]) -> List[int]:
    dynamic = tf.shape(tensor)
    if tensor.shape == tf.TensorShape(None):
        return dynamic
    static = tensor.shape.as_list()
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


# PrimaryCapsules layer in the original paper
class ConvCapsule(tf.keras.layers.Layer):
    def __init__(self, channels: int, capsule_dim: int, kernel_size: int, strides: Union[int, List[int], Tuple[int]], padding: str='valid', **kwargs):
        super(ConvCapsule, self).__init__(**kwargs)
        self.channels = channels
        self.capsule_dim = capsule_dim
        self.capsules = tf.keras.layers.Conv2D(capsule_dim*channels, kernel_size, strides, padding, **kwargs)

    def build(self, input_shape):
        self.capsules.build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = self.capsules(inputs, **kwargs)
        # (batch_size, height, width, filters): (batch_size, 6, 6, 256)

        outputs = tf.transpose(tf.reshape(outputs, shape_list(outputs)[:-1]+[self.channels, self.capsule_dim]), perm=[0, 3, 1, 2, 4])
        # (batch_size, channels, height, width, capsule_dim): (batch_size, 32, 6, 6, 8)

        outputs = tf.square(tf.norm(outputs, axis=-1, keepdims=True))/(1+tf.square(tf.norm(outputs, axis=-1, keepdims=True))) * outputs / tf.norm(outputs, axis=-1, keepdims=True)
        # squash

        return outputs


# DigitCaps layer in the original paper
class RoutingCapsule(tf.keras.layers.Layer):
    def __init__(self, num_class: int, dims: int, routing_iter: int=3, **kwargs):
        super(RoutingCapsule, self).__init__(**kwargs)
        self.num_class = num_class
        self.output_dim = dims
        self.routing_iter = routing_iter

    def build(self, input_shape):
        input_dims = input_shape[-1]
        self.caps_num = np.prod(input_shape[1:-1])
        self.weight = self.add_weight(name='weight',
                                      shape=(self.caps_num, self.num_class, input_dims, self.output_dim),
                                      initializer='GlorotUniform',
                                      trainable=True)  # (1152, 10, 8, 16)

    def call(self, inputs, **kwargs):
        # input size: (batch_size, channels, height, width, capsule_dim): (batch_size, 32, 6, 6, 8)

        outputs = tf.reshape(inputs, (-1, self.caps_num, 1, shape_list(inputs)[-1], 1))
        # (batch_size, channels*height*width (number of capsules), capsule_dim): (batch_size, 1152, 1, 8, 1)
        # put all capsules in parallel

        outputs = tf.repeat(outputs, self.num_class, axis=2)
        outputs = tf.repeat(outputs, self.output_dim, axis=-1)
        # (batch_size, # capsules, num_class, capsule_dim): (batch_size, 1152, 10, 8, 16)
        # ready for matrix multiplication

        outputs = tf.multiply(outputs, self.weight)
        # (batch_size, # capsules, num_class, capsule_dim, output_dim): (batch_size, 1152, 10, 8, 16)
        outputs = tf.reduce_sum(outputs, axis=-2)
        # (batch_size, # capsules, num_class, output_dim): (batch_size, 1152, 10, 16)
        # ready for routing

        b = tf.zeros((1, self.caps_num, self.num_class, 1))  # (1, 1152, 10, 1)

        for _ in range(self.routing_iter):
            c = tf.nn.softmax(b, axis=-1)
            # (1, # capsules, num_class, 1): (1, 1152, 10, 1)
            # computed softmax, extra dimensions of 1 are for broadcasting

            s = tf.reduce_sum(tf.multiply(outputs, c), axis=1, keepdims=True)
            # (batch_size, 1, num_class, output_dim): (batch_size, 1, 10, 16)
            # computed weighted sum

            v = tf.square(tf.norm(s, axis=-1, keepdims=True))/(1+tf.square(tf.norm(s, axis=-1, keepdims=True))) * s / tf.norm(s, axis=-1, keepdims=True)
            # squash

            if _ < self.routing_iter-1:
                update = tf.multiply(v, outputs)
                update = tf.reduce_mean(tf.reduce_sum(update, axis=-1, keepdims=True), axis=0, keepdims=True)
                b = b + update

            else:
                outputs = tf.reshape(v, (-1, self.num_class, self.output_dim))

        return outputs


class ReconstructionDecoder(tf.keras.layers.Layer):
    def __init__(self, dims: Union[List[int], Tuple[int]], original_img: Union[List[int], Tuple[int]], **kwargs):
        super(ReconstructionDecoder, self).__init__(**kwargs)
        self.denses = tf.keras.Sequential()
        for d in dims:
            self.denses.add(tf.keras.layers.Dense(d, activation='relu'))
        self.denses.add(tf.keras.layers.Dense(np.prod(original_img), activation='sigmoid'))
        self.original_img = list(original_img)

    def build(self, input_shape):
        self.denses.build((None, input_shape[-1]*input_shape[-2]))

    def call(self, inputs, **kwargs):
        # input size: (batch_size, num_class, output_dim): (batch_size, 10, 16)

        indices = tf.norm(inputs, axis=-1)
        indices = tf.argmax(indices, axis=-1)
        # (batch_size, num_class): (batch_size,)

        outputs = tf.expand_dims(tf.one_hot(indices, shape_list(inputs)[1]), axis=-1)
        # (batch_size, output_dim, 1): (batch_size, 10, 1)

        outputs = outputs * inputs
        # (batch_size, num_class, output_dim): (batch_size, 10, 16)

        outputs = tf.reshape(outputs, (shape_list(inputs)[0], -1))
        # (batch_size, num_class * output_dim): (batch_size, 160)

        outputs = tf.reshape(self.denses(outputs), [-1]+self.original_img)
        # (batch_size, ) + original_image: (batch_size, 28, 28, 1)

        return outputs


class MNISTCapsNet(tf.keras.Model):
    def __init__(self, conv: Optional[tf.keras.layers.Conv2D]=None, conv_cap: Optional[ConvCapsule]=None, digit_cap: Optional[RoutingCapsule]=None, reconstruct: Optional[ReconstructionDecoder]=None, **kwargs):
        super(MNISTCapsNet, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(256, 9, 1, activation='relu') if conv is None else conv
        self.conv_cap = ConvCapsule(32, 8, 9, 2, 'valid') if conv_cap is None else conv_cap
        self.digit_cap = RoutingCapsule(10, 16, 3) if digit_cap is None else digit_cap
        self.reconstruct = ReconstructionDecoder([512, 1024], [28, 28, 1]) if reconstruct is None else reconstruct

    def call(self, inputs, **kwargs):
        # (batch_size,) + original_image
        outputs = self.conv(inputs)
        outputs = self.conv_cap(outputs)
        class_vectors = self.digit_cap(outputs)
        recons_images = self.reconstruct(class_vectors)
        return class_vectors, recons_images

    def compute_loss(self, class_vectors: tf.Tensor, recons_images: tf.Tensor, labels: tf.Tensor, original_images: tf.Tensor, neg_scaling: float=0.5, recons_scaling: float=0.0005):
        """
        compute loss function
        :param class_vectors: prediction of classification in vectors, (batch_size, 10, 16) by default
        :param recons_images: reconstructed images, (batch_size, 28, 28, 1)
        :param labels: correct labels, (batch_size,)
        :param original_images: original input images, (batch_size, 28, 28, 1)
        :return: total loss, classification loss, reconstruction loss
        """
        logits = tf.norm(class_vectors, axis=-1)
        # (batch_size, 10)
        positive = tf.one_hot(labels, shape_list(class_vectors)[-2])
        # (batch_size, 10)

        class_loss = tf.reduce_sum(positive * tf.maximum(0., 0.9-logits) + neg_scaling * (1-positive) * tf.maximum(0., logits-0.1), axis=-1)
        # computed classification loss function, (batch_size,)

        difference = tf.reshape(recons_images-tf.cast(original_images, tf.float32)/255., (shape_list(class_vectors)[0], -1))
        # (batch_size, 784)

        recons_loss = tf.reduce_sum(tf.square(difference), axis=-1) * recons_scaling
        # computed reconstruction loss function, (batch_size,)

        class_loss = tf.reduce_mean(class_loss)
        recons_loss = tf.reduce_mean(recons_loss)

        return class_loss + recons_loss  #, class_loss, recons_loss
