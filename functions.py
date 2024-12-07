import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import json

####### 1 - functions
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels
  
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

def get_metadata(FOLDER,RANDOM_STATE=24):
    """
    This function takes a path to a FOLDER with has the structure 
    FOLDER
    - class
        -- element

    And returns an array with shape (NUM_CLASES, 2)
    [['class1', 'file_name_1'],
     ['class1', 'file_name_2'],
     ...
     ] 
    """

    # get array of [folder, file]
    SPECIES = []
    AUDIOS = []

    for folder in os.listdir(FOLDER):
        LISTDIR = os.listdir(os.path.join(FOLDER,folder))
        SPECIES.extend([folder] * len(LISTDIR))
        AUDIOS.extend(LISTDIR)

    metadata = np.column_stack((SPECIES,AUDIOS))

    return metadata

def get_missing_dictionary(FOLDER,RANDOM_STATE=24):
    """
    
    """ 
    DICT = {i:os.listdir(os.path.join(FOLDER, i)) for i in os.listdir(FOLDER)}

    return DICT

def get_species(FOLDER):
   return os.listdir(FOLDER)

def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

### 2 - SNR filter related
def power_of_signal(signal):
    return sum([i**2 for i in signal]) / len(signal)

def SNR(signal, noise):
    return np.log10(power_of_signal(signal) / power_of_signal(noise))

def SNR_noise(signal,noise_power):
    return np.log10(power_of_signal(signal) / noise_power)

# 2.1 some processing folder related
def read_species(base_dir = ""):
  PATH = os.path.join(base_dir,'species_64.txt')
  with open(PATH, 'r') as file:
    # Convert each line to an integer and store in a list
    return [line.strip() for line in file]


### 3 - plot format related
def plot_timeseries(signal, time_vector, width=0.025):
   # plot signal
  plt.figure(figsize=(12,6))
  plt.plot(time_vector, signal, linewidth=width, color="black")

  # Get the current axis
  ax = plt.gca()

  # Remove the right and upper spines
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  #plt.title("Sampled Musical Signal")
  plt.xlabel("duración (s)", fontsize=12)
  plt.ylabel("amplitud []", fontsize=12)
  #plt.title(NAME_Species)

  # Set the font size for tick labels on both axes
  plt.tick_params(axis='both', labelsize=10)  # Adjust the font size as needed

  plt.show()


def plot_spectrogram(spectrogram_db):
   pass


#### 4-SpectrogramGenerator related
def split_data(data_dir, test_size=0.2, random_state=42):
    # Get all class directories
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    train_files = []
    test_files = []

    for class_dir in class_dirs:
        # Get all files in class directory
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.txt')]
        
        # Split into train and test
        train, test = train_test_split(files, test_size=test_size, random_state=random_state)
        train_files.extend(train)
        test_files.extend(test)

    return train_files, test_files

class SpectrogramDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, shuffle=True, target_size=(128, 63), classes=None):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_size = target_size
        self.classes = classes
        self.filenames, self.labels = self._scan_directory()
        self.on_epoch_end()

    def _scan_directory(self):
        # Scan the directory to find all files and labels
        filenames = []
        labels = []
        class_names = self.classes or sorted(os.listdir(self.directory))
        
        for class_index, class_name in enumerate(class_names):
            class_folder = os.path.join(self.directory, class_name)
            class_files = os.listdir(class_folder)
            for file in class_files:
                if file.endswith('.txt'):
                    filenames.append(os.path.join(class_folder, file))
                    labels.append(class_index)
        
        return filenames, np.array(labels)

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        batch_filenames = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self._load_file(file) for file in batch_filenames]
        batch_data = np.array(batch_data)
        
        return batch_data, tf.keras.utils.to_categorical(batch_labels, num_classes=len(self.classes))

    def _load_file(self, filepath):
        # Load the .txt file and return it as a numpy array with target shape
        spectrogram = np.loadtxt(filepath)
        return np.expand_dims(spectrogram, axis=-1)

    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch
        if self.shuffle:
            indices = np.arange(len(self.filenames))
            np.random.shuffle(indices)
            self.filenames = [self.filenames[i] for i in indices]
            self.labels = self.labels[indices]