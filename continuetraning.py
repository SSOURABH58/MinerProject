import os
import re
import sys
import time
from typing import Any

from tensorflow.keras.callbacks import ModelCheckpoint ,LearningRateScheduler
from keras_preprocessing.image import ImageDataGenerator

import numpy as np

from tensorflow.keras import optimizers

MODEL_PATH = './data/00000-test-train/model-03-0.00.hdf5'
OPT_PATH = './data/00000-test-train/model-03-0.00.pkl'

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

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


def load_model_data(model_path, opt_path):
    model = load_model(model_path)
    with open(opt_path, 'rb') as fp:
      d = pickle.load(fp)
      epoch = d['epoch']
      opt = d['opt']
      return epoch, model, opt

epoch, model, opt = load_model_data(MODEL_PATH, OPT_PATH)

outdir = "./data/"
run_desc = "test-train"
batch_size = 8
num_classes = 10
img_height, img_width = 256,256
learning_rate = 5e-5

# note: often it is not necessary to recompile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam.from_config(opt),
    metrics=['accuracy'])

outdir = "./data/"
run_desc = "cont-train"
num_classes = 10

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
  # input_shape, x_train, y_train, x_test, y_test = obtain_data()
  # train_model(model, initial_epoch=epoch, max_epochs=6)

  nb_categories, train_generator, val_generator = obtain_data()
  # model = build_model(nb_categories)
  train_model(model, initial_epoch=epoch,max_epochs=100)