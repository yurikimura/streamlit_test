import streamlit as st
import numpy as np
import os
import random
import librosa
import warnings
from keras.models import load_model

from datetime import timedelta

reconstructed_model = load_model("vtuber_reco.h5")
target_label = {0:"Calliope",1:"Ninomae",2:"Watson",3:"Gura",4:"Kiara"}

import streamlit.components.v1 as components

warnings.simplefilter('ignore')

sample_rate = 44100
threshold = 20
sample_length = 7680
batch_size = 16
epoch = 50
time = 3

def preprocess(audio,threshold,sample_length,sample_rate):
    audio, _ = librosa.effects.trim(audio, threshold)

    # すべての音声ファイルを指定した同じサイズに変換
    if threshold is not None:
        if len(audio) <= sample_length:
            # padding
            pad = sample_length - len(audio)
            audio = np.concatenate((audio, np.zeros(pad, dtype=np.float32)))
        else:
            # trimming
            start = random.randint(0, len(audio) - sample_length - 1)
            audio = audio[start:start + sample_length]
        stft = np.abs(librosa.stft(audio))
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40),axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
        mel = np.mean(librosa.feature.melspectrogram(audio, sr=sample_rate),axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate),axis=1)

        feature = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature = np.expand_dims(feature, axis=1)

    return feature

def predict_timestamp_and_remove(num_cut):
    time_from = str(timedelta(seconds=0))
    last_speaker = None

    for i in range(num_cut):
        time_to = str(timedelta(seconds=(i+1)*time))
        x_batch = []  # feature
        path = f'{i}.wav'
        audio, _ = librosa.load(path, sr=sample_rate)
        mfccs = preprocess(audio,threshold,sample_length,sample_rate)
        x_batch.append(mfccs)
        x_batch = np.asarray(x_batch)
        pred = reconstructed_model.predict(x_batch,verbose=0)

        if pred.argmax() == last_speaker:
            pass
        # the first
        elif last_speaker == None:
            last_speaker = pred.argmax()
        # the last
        elif i+1 == num_cut:
            # target_timestamp = [f'{time_from} --> {time_to}']
            # with server_state_lock.count:
            #     server_state.df.loc[len(df)] = target_timestamp+list(pred)
            st.write(target_label[pred.argmax()])
            st.write(f'probability: {round(max(pred[0]),2)}')
        else:
            st.write(f'{time_from} --> {time_to}')
            st.write(target_label[pred.argmax()])
            st.write(f'probability: {round(max(pred[0]),2)}')
            # target_timestamp = [f'{time_from} --> {time_to}']
            # with server_state_lock.count:
            #     server_state.df.loc[len(df)] = target_timestamp+pred.tolist()[0]
            # last_speaker = pred.argmax()
            # time_from = time_to

        #st.write(str(pred))
        os.remove(path)
    return

n=10
predict_timestamp_and_remove(n)
