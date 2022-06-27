import os
import re
import sys
import time
import numpy as np
from typing import Any, List, Tuple, Union

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, \
    LearningRateScheduler, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
import pickle

from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models, Model, optimizers

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

def generate_output_dir(outdir, run_desc):
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir( \
            os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(run_dir)
    os.makedirs(run_dir)
    return run_dir


# From StyleGAN2
class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and
    optionally force flushing on both stdout and the file."""

    def __init__(self, file_name: str = None, file_mode: str = "w", \
                 should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, \
                 traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove
            stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def obtain_data():
    data = "testdata"

    category_names = sorted(os.listdir(data))
    nb_categories = len(category_names)  # number of category --

    # only rescaling
    datagen = ImageDataGenerator(
        rescale=1. / 255, validation_split=0.2
    )
    # these are generators for train/test data that will read pictures #found in the defined subfolders of 'data/'
    print('Total number of train images :')

    train_generator = datagen.flow_from_directory(
        directory=data,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="training",
        class_mode="categorical")

    print('Total number of validation images :')

    val_generator = datagen.flow_from_directory(
        directory=data,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        subset="validation",
        class_mode="categorical")

    return nb_categories,train_generator,val_generator


outdir = "./data/"
run_desc = "test-train"
batch_size = 8
num_classes = 10
img_height, img_width = 256,256
learning_rate = 5e-5

# nb_categories,category_names = obtain_data()

run_dir = generate_output_dir(outdir, run_desc)
print(f"Results saved to: {run_dir}")

class MyModelCheckpoint(ModelCheckpoint):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    super().on_epoch_end(epoch,logs)\

    # Also save the optimizer state
    filepath = self._get_file_path(epoch=epoch,
        logs=logs)
    filepath = filepath.rsplit( ".", 1 )[ 0 ]
    filepath += ".pkl"

    with open(filepath, 'wb') as fp:
      pickle.dump(
        {
          'opt': model.optimizer.get_config(),
          'epoch': epoch+1
         # Add additional keys if you need to store more values
        }, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print('\nEpoch %05d: saving optimizaer to %s' % (epoch + 1, filepath))

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)

def build_model( num_classes):
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=tf.keras.optimizers.Adam(),
    #     metrics=['accuracy'])
    img_height, img_width = 256, 256
    conv_base = vgg16.VGG16(weights='imagenet', include_top=False, pooling='max',
                            input_shape=(img_width, img_height, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate, clipnorm=1.),
                  metrics=['accuracy'])

    return model

def train_model(model, initial_epoch=0, max_epochs=10):
    start_time = time.time()

    checkpoint_cb = MyModelCheckpoint(
        os.path.join(run_dir, 'model-{epoch:02d}-{val_loss:.2f}.hdf5'),
        monitor='val_loss',verbose=1)

    lr_sched_cb = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, \
                                      step_size=2)
    cb = [checkpoint_cb, lr_sched_cb]

    # history = model.fit(train_generator,
    #                     epochs=epochs,
    #                     shuffle=False,
    #                     validation_data=val_generator,
    #                     callbacks=[checkpoint]
    #                     )

    model.fit(train_generator,
              batch_size=batch_size,
              epochs=max_epochs,
              initial_epoch = initial_epoch,
              verbose=2, callbacks=cb,
              validation_data=val_generator)
    score = model.evaluate(train_generator, verbose=0, callbacks=cb)
    print('Test loss: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))

    elapsed_time = time.time() - start_time
    print("Elapsed time: {}".format(hms_string(elapsed_time)))

with Logger(os.path.join(run_dir, 'log.txt')):
    nb_categories,train_generator,val_generator = obtain_data()
    model = build_model(nb_categories)
    train_model(model, max_epochs=3)