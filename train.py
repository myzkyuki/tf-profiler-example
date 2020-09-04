import os
import math
import argparse
import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging


def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label


def build_dataset(split, batch_size, optimize_dataset):
    dataset, ds_info = tfds.load('mnist', split=split,
                                 as_supervised=True,
                                 with_info=True)
    if optimize_dataset:
        dataset = dataset.map(normalize_image,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(normalize_image)

    dataset = dataset.batch(batch_size)
    if optimize_dataset:
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if split == 'train':
        dataset = dataset.repeat()
        dataset = dataset.shuffle(100)

    num_examples = ds_info.splits[split].num_examples
    return dataset, num_examples



def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax', dtype='float32')
    ])
    return model


def train_with_fit(model, optimizer, loss_fn,
                   epochs, steps_per_epoch, ds_train, ds_test,
                   log_dir, profile_start_step, profile_end_step):
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, profile_batch=[profile_start_step,profile_end_step])
    model.fit(ds_train,
              epochs=epochs,
              callbacks=[tboard_callback],
              validation_data=ds_test,
              steps_per_epoch=steps_per_epoch)


def train_with_custom_loop(model, optimizer, loss_fn,
                           epochs, steps_per_epoch, ds_train, ds_test,
                           log_dir, profile_start_step, profile_end_step):
    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def train_step(X, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = loss_fn(y_true, y_pred)
        graidents = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(graidents, model.trainable_weights))

        train_loss.update_state(loss)
        train_acc.update_state(y_true, y_pred)

        return loss

    @tf.function
    def validation_step(X, y_true):
        y_pred = model(X)
        loss = loss_fn(y_true, y_pred)

        val_loss.update_state(loss)
        val_acc.update_state(y_true, y_pred)

        return loss

    global_step = optimizer.iterations.numpy()
    summary_writer = tf.summary.create_file_writer(log_dir)
    train_iter = iter(ds_train)
    total_steps = epochs * steps_per_epoch
    logging_interval = math.ceil(steps_per_epoch / 20)

    with summary_writer.as_default():
        for global_step in range(global_step, total_steps):
            if global_step == profile_start_step:
                tf.profiler.experimental.start(log_dir)
                logging.info(f'Start profile at {global_step}')
            elif global_step == profile_end_step:
                tf.profiler.experimental.stop()
                logging.info(f'End profile at {global_step}')

            with tf.profiler.experimental.Trace('train',
                                                step_num=global_step, _r=1):
                X, y_true = next(train_iter)
                train_step(X, y_true)

                if (global_step + 1) % logging_interval == 0:
                    logging.info(f'Steps: {global_step}, '
                                 f'Train Acc: {train_acc.result():.3f}, '
                                 f'Train Loss: {train_loss.result():.3f}')

                    tf.summary.scalar(
                        'Train/Acc', data=train_acc.result(), step=global_step)
                    tf.summary.scalar(
                        'Train/Loss', data=train_loss.result(),
                        step=global_step)
                    train_loss.reset_states()
                    train_acc.reset_states()

                if ((global_step + 1) % steps_per_epoch == 0 or
                    global_step == total_steps - 1):
                    for X, y_true in ds_test:
                        validation_step(X, y_true)

                    logging.info(f'Steps: {global_step}, '
                                 f'Val Acc: {val_acc.result():.3f}, '
                                 f'Val Loss: {val_loss.result():.3f}')

                    tf.summary.scalar(
                        'Val/Acc', data=val_acc.result(), step=global_step)
                    tf.summary.scalar(
                        'Val/Loss', data=val_loss.result(), step=global_step)

                    val_loss.reset_states()
                    val_acc.reset_states()


def log_args(args):
    for key, val in vars(args).items():
        logging.info(f'{key}: {val}')

def main(args):
    log_args(args)

    if args.gpu_private:
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    batch_size = args.batch_size

    ds_train, num_train = build_dataset(split='train', batch_size=batch_size,
                                        optimize_dataset=args.optimize_dataset)
    ds_test, num_test = build_dataset(split='test', batch_size=batch_size,
                                      optimize_dataset=args.optimize_dataset)

    steps_per_epoch = num_train // batch_size
    profile_start_step = int(steps_per_epoch * 1.5)
    profile_end_step = profile_start_step + args.profile_steps

    if not args.custom_train_loop and args.mixed_precision:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    model = build_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    log_dir = os.path.join(args.log_dir,
                           datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    if args.custom_train_loop:
        train_fn = train_with_custom_loop
    else:
        train_fn = train_with_fit

    train_start = datetime.datetime.now()
    train_fn(model, optimizer, loss_fn,
             args.epochs, steps_per_epoch, ds_train, ds_test,
             log_dir, profile_start_step, profile_end_step)
    train_sec = datetime.datetime.now() - train_start
    logging.info(f'Train sec: {train_sec}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='mini batch size')
    parser.add_argument('--profile_steps', type=int, default=20,
                        help='number of steps to profile')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='profile log directory')
    parser.add_argument('--optimize_dataset', action='store_true',
                        help='whether to optimize the dataset pipeline')
    parser.add_argument('--gpu_private', action='store_true',
                        help='whether the GPU uses a dedicated thread')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='whether to use mixed precision')
    parser.add_argument('--custom_train_loop', action='store_true',
                        help='whether to use custom train loop')
    args = parser.parse_args()
    logging.set_verbosity(logging.INFO)
    main(args)
