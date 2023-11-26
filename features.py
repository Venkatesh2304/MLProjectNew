from concurrent.futures import ProcessPoolExecutor
import librosa
import numpy as np
import os
import csv
import pandas as pd
from tqdm import tqdm, trange
from glob import glob 

def extract_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)

    # Spectral features
    sp_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    meanfreq = np.mean(sp_cent)
    sd = np.std(sp_cent)
    median = np.median(sp_cent)
    Q25 = np.percentile(sp_cent, 25)
    Q75 = np.percentile(sp_cent, 75)
    IQR = Q75 - Q25
    sfm = np.mean(librosa.feature.spectral_flatness(y=y))
    pitches = librosa.pyin(y=y,fmin=30,fmax=300)[0]
    pitches = pitches[pitches>0]
    if(len(pitches)==0):
        return 0
    meanfun = np.mean(pitches)
    #print(meanfun,audio_path)
    minfun = np.min(pitches)
    maxfun = np.max(pitches)

    # Dominant frequency features
    mel_freq = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_freq = librosa.feature.spectral_centroid(S=mel_freq)
    meandom = np.mean(mel_freq)
    mindom = np.min(mel_freq)
    maxdom = np.max(mel_freq)
    dfrange = maxdom - mindom

    # Modulation index
    modindx = np.mean(librosa.feature.mfcc(y=y, sr=sr))

    features = [meanfreq,sd,median,Q25,Q75,IQR,sfm,meanfun,minfun,maxfun,meandom,mindom,maxdom,dfrange,modindx]
    return features
 
import sys 
names= { "vinu" : 0 , "ven" : 1 , "aad" : 2 }
v = names[ str(sys.argv[1]) ]

data = ["path","meanfreq","sd","median","Q25","Q75","IQR","sfm","meanfun","minfun","maxfun","meandom","mindom","maxdom","dfrange","modindx"]
files = glob("filtered_data/*")

size = len(files)//len(names) 

files.sort()
# files = files[ size*v : size*(v+1) ]
result = []

for s in trange(0,len(files),1) :
    a = extract_features(files[s])
    if a != 0 :
       result.append( [ files[s] ] + a  )

pd.DataFrame( result , columns = data ).to_csv(f"features_removed.csv",index=False)
