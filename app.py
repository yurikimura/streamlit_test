import streamlit as st
import numpy as np
import os
import random
import librosa
import warnings
from keras.models import load_model

import wave
import struct
import math
from datetime import timedelta

warnings.simplefilter('ignore')

sample_rate = 44100
threshold = 20
sample_length = 7680
batch_size = 16
epoch = 50
time = 3

reconstructed_model = load_model("vtuber_reco.h5")
predfile = st.file_uploader("Upload file", type=['wav','mp3','mp4'])
a = st.text_input(label="OR YOUTUBE LINK HERE👇:")
print(a)

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

def pred_with_timestamp(pred_path, time):
    wr = wave.open(pred_path, 'r')

    #waveファイルが持つ性質を取得
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate()
    fn = wr.getnframes()
    total_time = 1.0 * fn / fr
    integer = math.floor(total_time)
    t = int(time)
    frames = int(ch * fr * t)
    num_cut = int(integer//t)

    # 確認用
    st.write("total time(s) : ", total_time)
    st.write("total time(integer) : ", integer)
    st.write("")

    # waveの実データを取得し数値化
    data = wr.readframes(wr.getnframes())
    wr.close()
    X = np.frombuffer(data, dtype='int16')

    for i in range(num_cut):
        #print(str(i) + ".wav --> OK!")
        #出力データを生成
        outf = str(i) + '.wav'
        start_cut = i*frames
        end_cut = i*frames + frames
        Y = X[start_cut:end_cut]
        outd = struct.pack("h" * len(Y), *Y)

        # 書き出し
        ww = wave.open(outf, 'w')
        ww.setnchannels(ch)
        ww.setsampwidth(width)
        ww.setframerate(fr)
        ww.writeframes(outd)
        ww.close()

    time_to = str(timedelta(seconds=0))
    for i in range(num_cut):
        time_from = time_to
        time_to = str(timedelta(seconds=(i+1)*time))
        st.write(f'{time_from} --> {time_to}')
        x_batch = []  # feature
        path = f'{i}.wav'
        audio, _ = librosa.load(path, sr=sample_rate)
        mfccs = preprocess(audio,threshold,sample_length,sample_rate)
        x_batch.append(mfccs)
        x_batch = np.asarray(x_batch)
        pred = reconstructed_model.predict(x_batch,verbose=0)
        st.write(pred.argmax())
        #st.write(str(pred))
        os.remove(path)

    return

try:
    pred_with_timestamp(predfile,3)
except:
    pass
