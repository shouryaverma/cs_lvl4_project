import wfdb as wf
import numpy as np
from biosppy.signals import ecg
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import glob
from wfdb import processing


def extract_data():
    data_files = glob.glob('./mit-bih/*.atr')
    data_files = [i[:-4] for i in data_files]
    data_files.sort()
    return data_files

files = extract_data()
i=0

datfile = files[i]
record = wf.rdsamp(datfile)
ann = wf.rdann(datfile, 'atr')