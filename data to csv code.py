import wfdb as wf
import numpy as np
import glob
from matplotlib import pyplot as plt
from biosppy.signals import ecg
from scipy import signal

def extract_data():
    data_files = glob.glob('./mit-bih/*.atr')
    data_files = [i[:-4] for i in data_files]
    data_files.sort()
    return data_files

records = extract_data()
print('Total files: ', len(records))

# Instead of using the annotations to find the beats, we will
# use R-peak detection instead. The reason for this is so that
# the same logic can be used to analyze new and un-annotated
# ECG data. We use the annotations here only to classify the
# beat as either Normal or Abnormal and to train the model.
# Reference:
# https://physionet.org/physiobank/database/html/mitdbdir/intro.htm

good_beats = ['N','L','R','B','A','a','J','S','V','r',
             'F','e','j','n','E','/','f','Q','?']

# Loop through each input file. Each file contains one
# record of ECG readings, sampled at 360 readings per
# second.


for path in records:
    pathpts = path.split('/')
    fn = pathpts[-1]
    print('Loading file:', path)

    # Read in the data
    record = wf.rdsamp(path)
    annotation = wf.rdann(path, 'atr')

    # Print some meta informations
    print('    Sampling frequency used for this record:', record[1].get('fs'))
    print('    Shape of loaded data array:', record[0].shape)
    print('    Number of loaded annotations:', len(annotation.num))
    
    # Get the ECG values from the file.
    data = record[0].transpose()

    # Generate the classifications based on the annotations.
    # 0.0 = undetermined
    # 1.0 = normal
    # 2.0 = LBBBB
    # 3.0 = RBBBB
    # 4.0 = Premature Ventricular contraction
    # 5.0 = Atrial Premature beat
    # 6.0 = Fusion ventricular normal beat
    # 7.0 = Fusion of paced and normal beat
    # 8.0 = paced beat
    
    clas = np.array(annotation.symbol)
    rate = np.zeros_like(clas, dtype='float')
    for clasid, clasval in enumerate(clas):
        if (clasval == 'N'):
            rate[clasid] = 1.0 # Normal
        elif (clasval == 'L'):
            rate[clasid] = 2.0 # LBBBB
        elif (clasval == 'R'):
            rate[clasid] = 3.0 # RBBBB
        elif (clasval == 'V'):
            rate[clasid] = 4.0 # Premature Ventricular contraction
        elif (clasval == 'A'):
            rate[clasid] = 5.0 # Atrial Premature beat
        elif (clasval == 'F'):
            rate[clasid] = 6.0 # Fusion ventricular normal beat
        elif (clasval == 'f'):
            rate[clasid] = 7.0 # Fusion of paced and normal beat
        elif (clasval == '/'):
            rate[clasid] = 8.0 # paced beat
            
    rates = np.zeros_like(data[0], dtype='float')
    rates[annotation.sample] = rate
    
    indices = np.arange(data[0].size, dtype='int')

    # Process each channel separately (2 per input file).
    for channelid, channel in enumerate(data):
        chname = record[1].get('sig_name')[channelid]
        print('    ECG channel type:', chname)
        
        # Find rpeaks in the ECG data. Most should match with
        # the annotations.
        out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
        rpeaks = np.zeros_like(channel, dtype='float')
        rpeaks[out['rpeaks']] = 1.0
        
        beatstoremove = np.array([0])

        # Split into individual heartbeats. For each heartbeat
        # record, append classification (normal/abnormal).
        beats = np.split(channel, out['rpeaks'])
        for ind, ind_val in enumerate(out['rpeaks']):
            beat_start = ind == 0
            beat_end = ind == len(beats) - 1

            # Skip start and end beat.
            if (beat_start or beat_end):
                continue

            # Get the classification value that is on
            # or near the position of the rpeak index.
            from_ind = 0 if ind_val < 10 else ind_val - 10
            to_ind = ind_val + 10
            clasval = rates[from_ind:to_ind].max()
            
            # Skip beat if there is no classification.
            if (clasval == 0.0):
                beatstoremove = np.append(beatstoremove, ind)
                continue

            # Append some extra readings from next beat.
            beats[ind] = np.append(beats[idx], beats[ind+1][:40])

            # Normalize the readings to a 0-1 range for ML purposes.
            beats[ind] = (beats[ind] - beats[ind].min()) / beats[ind].ptp()

            # Resample from 360Hz to 125Hz
            newsize = int((beats[ind].size * 125 / 360) + 0.5)
            beats[ind] = signal.resample(beats[ind], newsize)

            # Skipping records that are too long.
            if (beats[ind].size > 187):
                beatstoremove = np.append(beatstoremove, ind)
                continue

            # Pad with zeroes.
            zerocount = 187 - beats[ind].size
            beats[ind] = np.pad(beats[ind], (0, zerocount), 'constant', constant_values=(0.0, 0.0))

            # Append the classification to the beat data.
            beats[ind] = np.append(beats[ind], clasval)

        beatstoremove = np.append(beatstoremove, len(beats)-1)

        # Remove first and last beats and the ones without classification.
        beats = np.delete(beats, beatstoremove)

        # Save to CSV file.
        savedata = np.array(list(beats[:]), dtype=np.float)
        outfn = './'+fn+'_'+chname+'.csv'
        print('    Generating ', outfn)
        with open(outfn, "wb") as fin:
            np.savetxt(fin, savedata, delimiter=",", fmt='%f')
