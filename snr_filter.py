import numpy as np
import librosa
import os
from functions import power_of_signal, SNR, SNR_noise
from functions import plot_timeseries

# Read the list from the text file
def read_species_file(PATH=""):
    with open(PATH, 'r') as file:
        # Convert each line to an integer and store in a list
        VALID_SPECIES = [line.strip() for line in file]
    return VALID_SPECIES

BASE = "datos"

class SNR_filter():
    def __init__(self, BASE_PATH=BASE, species="brnhao1", item="XC19607.ogg") -> None:
        #assert species not in VALID_SPECIES, "Invalid name of species, it has to be any of: {}".format(VALID_SPECIES)
        self.path = os.path.join(BASE_PATH,species,item)
        self.base = BASE_PATH
        self.species = species
        self.item = item
        self.segments = []

        self.audio, self.sr = librosa.load(self.path,sr=32000)
        self.time_vector = [i/self.sr for i in range(len(self.audio))]
        
        #def create_spectrograms(self):
        self.spectrogram = librosa.feature.melspectrogram(y=self.audio,sr=self.sr)
        self.spectrogram_db = librosa.power_to_db(self.spectrogram, ref=np.max)
        self.times = librosa.frames_to_time(np.arange(self.spectrogram.shape[-1]), 
                                            sr=self.sr)

    def plot_signal(self):
        plot_timeseries(self.audio, self.time_vector)

    def filter(self):
        """
        Noise is considered to be the last 0.05 seconds of the audio
        Also the semgents are considered here to have that same length
        After maxima selection this duration is expanded.
        """
        noise_power = power_of_signal(self.audio[-1600:])
        powers = []
        for i in range(len(self.audio) // 1600):
            sub_signal = self.audio[i * 1600 : (i + 1) * 1600]
            powers.append(SNR_noise(sub_signal,noise_power))

        # sort the powers for the sake of finding maxima and consecutive maxima
        sorted_argmax = np.argsort(powers, axis=0, kind='quicksort')

        """
        some audios don't last enough for selecting several segments.
        the following snippet controls this situation
        """

        for index_ in [-1,-5, -9]:
            try:
                if int(sorted_argmax[index_]) * 1600 < 16000:
                    first_maxima_index = 16001
                elif int(sorted_argmax[index_]) * 1600 > len(self.time_vector) - 16000:
                    first_maxima_index = len(self.time_vector) - 16001
                else:
                    first_maxima_index = int(sorted_argmax[index_] * 1600)

                
                first_maxima_frame = self.time_vector[first_maxima_index - 16000]
                first_maxima_frame_ = self.time_vector[first_maxima_index + 16000]

                # Find the frame index closest to start time
                time_frame = (np.argmin(np.abs(self.times - first_maxima_frame)),
                            np.argmin(np.abs(self.times - first_maxima_frame_)))
                
                self.segments.append(time_frame)
            except:
                print("No se pudo: {}-{}".format(self.species,self.item))
                pass




class SNR_filter_():
    def __init__(self, BASE_PATH=BASE, species="brnhao1", item="XC19607.ogg") -> None:
        #assert species not in VALID_SPECIES, "Invalid name of species, it has to be any of: {}".format(VALID_SPECIES)
        self.path = os.path.join(BASE_PATH,species,item)
        self.base = BASE_PATH
        self.species = species
        self.item = item
        self.segments = []

        self.audio, self.sr = librosa.load(self.path,sr=32000)
        self.time_vector = [i/self.sr for i in range(len(self.audio))]
        
        #def create_spectrograms(self):
        #self.spectrogram = librosa.feature.melspectrogram(y=self.audio,sr=self.sr)
        #self.spectrogram_db = librosa.power_to_db(self.spectrogram, ref=np.max)
        #self.times = librosa.frames_to_time(np.arange(self.spectrogram.shape[-1]), 
        #                                    sr=self.sr)

    def plot_signal(self):
        plot_timeseries(self.audio, self.time_vector)

    def filter(self):
        """
        Noise is considered to be the last 0.05 seconds of the audio
        Also the semgents are considered here to have that same length
        After maxima selection this duration is expanded.
        """
        noise_power = power_of_signal(self.audio[-12000:])
        powers = []
        for i in range(len(self.audio) // 12000):
            sub_signal = self.audio[i * 12000 : (i + 1) * 12000]
            powers.append(SNR_noise(sub_signal,noise_power))

        # sort the powers for the sake of finding maxima and consecutive maxima
        sorted_argmax = np.argsort(powers, axis=0, kind='quicksort')

        """
        some audios don't last enough for selecting several segments.
        the following snippet controls this situation
        """
        for index_ in [-1,-5, -9]:
            #try:
            if int(sorted_argmax[index_]) * 12000 < 16000:
                first_maxima_index = 16001
            elif int(sorted_argmax[index_]) * 12000 > len(self.time_vector) - 16000:
                first_maxima_index = len(self.time_vector) - 16001
            else:
                first_maxima_index = int(sorted_argmax[index_] * 12000)
            
            first_maxima_frame = self.time_vector[first_maxima_index - 16000]
            first_maxima_frame_ = self.time_vector[first_maxima_index + 16000]

            self.segments.append((first_maxima_frame,first_maxima_frame_))

            # Find the frame index closest to start time
            time_frame = (np.argmin(np.abs(self.times - first_maxima_frame)),
                        np.argmin(np.abs(self.times - first_maxima_frame_)))
            
            #self.segments.append(time_frame)
            #except:
            #print("No se pudo: {}-{}".format(self.species,self.item))
            #pass
