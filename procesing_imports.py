import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split
import numpy as np
from functions import plot_timeseries, get_metadata, get_missing_dictionary
from functions import power_of_signal, SNR, SNR_noise, js_r
import matplotlib.pyplot as plt
from snr_filter import *
import librosa
from tqdm import tqdm
import json