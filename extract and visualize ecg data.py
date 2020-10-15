import wfdb as wf
import numpy as np
from matplotlib import pyplot as plt
from biosppy.signals import ecg
from scipy import signal

def extract_data():
    data_files = glob.glob('./mit-bih/*.atr')
    data_files = [i[:-4] for i in data_files]
    data_files.sort()
    return data_files

files = extract_data()


i=4 #file number 0-47

datfile = files[i]
record = wf.rdsamp(datfile)
ann = wf.rdann(datfile, 'atr')

channel_num=0 # There are 2 channels
samplestart =0 # Start of the sample in the file.
samplesize=5000 # Number of readings (360 per second) 
sampleend = samplestart + samplesize


record = wf.rdsamp(datfile)
ann = wf.rdann(datfile, 'atr')

# Get data and anns for the samples selected below.
channel = record[0][samplestart:sampleend, channel_num]

# Plot the heart beats. Time scale is number of readings
# divided by sampling frequency.
time_scale = (np.arange(samplesize, dtype = 'float') + samplestart) / record[1].get('fs')
plt.figure(figsize=(20,10))
plt.plot(time_scale, channel)

# Extract anns.
where = np.logical_and(ann.sample >= samplestart, ann.sample < sampleend)
anns = ann.sample[where] - samplestart
annotypes = np.array(ann.symbol)
annotypes = annotypes[where]

# Plot the anns.
annotimes = time_scale[anns]
plt.plot(annotimes, np.ones_like(annotimes) * channel.max() * 1.4, 'ro')

# ann codes.
for idx, annot in enumerate(anns):
    plt.annotate(annotypes[idx], xy = (time_scale[annot], channel.max() * 1.1))

plt.xlim([samplestart / record[1].get('fs'), (sampleend / record[1].get('fs')) + 1])
plt.xlabel('Offset')
plt.ylabel(record[1].get('sig_name')[channel_num])
plt.show()