import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from capsule import MNISTCapsNet

IMAGE_SIZE = (28, 28, 1)

parser = argparse.ArgumentParser(description='Capsule Network for MNIST')
parser.add_argument('--batch-size', type=int, default=32, help='batch size for training, validating and testing, 32 by default')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training, 100 by default')
parser.add_argument('--learning-rate', type=float, default=0.001, help='initial learning rate, 0.001 by default')
parser.add_argument('--lr-decay-rate', type=float, default=0.8, help='learning rate decay factor, 0.8 by default')
parser.add_argument('--lr-decay-steps', type=int, default=10000, help='steps to take until the next decay, 10000 by default')
parser.add_argument('--save-model', type=bool, default=True, help='whether to save the model or not, True by default')
parser.add_argument('--model-path', type=str, default='./model', help='path to save the trained model, ./model by default')
parser.add_argument('--gpu', type=str, default='0', help='ids of the GPUs to run the program on, use : to indicate a slice, 0 by default')
parser.add_argument('--testing', type=bool, default=False, help='for testing purposes, DO NOT use it')
parser.add_argument('--datasets', type=str, default='kmnist', help='change to other kind of datasets')

args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs

try:
    lower_gpu = int(args.gpu)
    upper_gpu = lower_gpu + 1
except ValueError:
    lower_gpu, upper_gpu = args.gpu.split(':')
    lower_gpu = int(lower_gpu)
    upper_gpu = int(upper_gpu)

gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_visible_devices(devices=gpus[lower_gpu:upper_gpu], device_type='GPU')

def augmentation():
    operations = tf.keras.Sequential()
    operations.add(tf.keras.layers.Rescaling(1./255))
    operations.add(tf.keras.layers.ZeroPadding2D(2))
    operations.add(tf.keras.layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]))

    def img_aug(x: tf.Tensor):
        return operations(x)

    return img_aug


def normalize():
    operations = tf.keras.layers.Rescaling(1./255)

    def img_norm(x: tf.Tensor):
        return operations(x)

    return img_norm


if __name__ == '__main__':
    train_set, valid_set, test_set = tfds.load(args.datasets, split=['train[:90%]', 'train[90%:]', 'test'], shuffle_files=True, as_supervised=True)
    train_set = train_set.shuffle(1024, reshuffle_each_iteration=True).prefetch(1024)
    per_step = len(train_set) // 32 + 1
    img_aug = augmentation()
    img_norm = normalize()
    model = MNISTCapsNet()

    if args.testing:
        dummy_input = np.zeros((1, 28, 28, 1), dtype=float)
        model(dummy_input, training=False)
        exit()


    @tf.function
    def training(x, y):
        with tf.GradientTape() as tape:
            class_vectors, recons_images = model(img_aug(x), training=True)
            total_loss = model.compute_loss(class_vectors, recons_images, y, x)
            grad = tape.gradient(total_loss, model.trainable_weights)
        opt.apply_gradients(zip(grad, model.trainable_weights))
        train_metric.update_state(y, tf.norm(class_vectors, axis=-1))
        return total_loss

    @tf.function
    def validation(x, y):
        class_vectors, recons_images = model(img_norm(x), training=False)
        total_loss = model.compute_loss(class_vectors, recons_images, y, x)
        valid_metric.update_state(y, tf.norm(class_vectors, axis=-1))
        return total_loss

    @tf.function
    def testing(x, y):
        class_vectors, recons_images = model(img_norm(x), training=False)
        total_loss = model.compute_loss(class_vectors, recons_images, y, x)
        test_metric.update_state(y, tf.norm(class_vectors, axis=-1))
        return total_loss

    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, args.lr_decay_steps, args.lr_decay_rate, staircase=True)
    opt = tf.keras.optimizers.Adam(lr_scheduler)
    train_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    valid_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs}')
        pb = tf.keras.utils.Progbar(per_step, stateful_metrics=['training_loss', 'training_metric'], verbose=1)

        for x, y in train_set.batch(batch_size):
            loss = training(x, y)
            pb.add(1, [('training_loss', float(loss)), ('training_metric', float(train_metric.result()))])

        losses = []
        for x, y in valid_set.batch(batch_size):
            loss = validation(x, y)
            losses.append(loss)
        print(f'valid_loss: {np.mean(losses):.4f}, valid_metric: {float(valid_metric.result()):.4f}\n')

        train_metric.reset_state()
        valid_metric.reset_state()

    if args.save_model:
        model.save(args.model_path)

    losses = []
    test_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in test_set.batch(batch_size):
        loss = testing(x, y)
        losses.append(loss)
    print(f'Test set loss: {np.mean(losses):.4f}, accuracy: {float(test_metric.result()):.4f}\n')
    
losses=np.array(losses)
plt.ylabel('losses')
plt.xlabel('batch_num')
plt.plot(losses)

