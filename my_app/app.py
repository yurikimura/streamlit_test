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

# playing audio file
import io
from scipy.io import wavfile

import streamlit.components.v1 as components

# import youtube_dl
# import ffmpeg

warnings.simplefilter('ignore')

sample_rate = 44100
threshold = 20
sample_length = 7680
batch_size = 16
epoch = 50
time = 3

reconstructed_model = load_model("vtuber_reco.h5")
target_label = {0:"Calliope",1:"Ninomae",2:"Watson",3:"Gura",4:"Kiara"}
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

predfile = st.sidebar.file_uploader("Upload file", type=['wav'])


# def youtube_to_wav(youtube_url):
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([youtube_url])
#         stream = ffmpeg.input('output.m4a')
#         stream = ffmpeg.output(stream, 'output.wav')
#         wr = wave.open('output.wav', 'r')
#     return wr

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

# def load_wave_file(pred_path):
#     try:
#         wr = wave.open(pred_path, 'r')
#     except:
#         st.write("preparing for prediction....")
#         stream = ffmpeg.input(Path(pred_path))
#         stream = ffmpeg.output(stream, "test.wav")
#         ffmpeg.run(stream)
#         wr = wave.open("test.wav", 'r')
#     return wr


def target_cropper(wr, time):
    #waveファイルが持つ性質を取得
    ch = wr.getnchannels()#モノラルorステレオ
    width = wr.getsampwidth()
    fr = wr.getframerate()#サンプリング周波数
    fn = wr.getnframes()#フレームの総数
    total_time = 1.0 * fn / fr
    integer = math.floor(total_time)
    t = int(time)
    frames = int(ch * fr * t)
    num_cut = int(integer//t)

    # 確認用
    st.write(f'Target File :  {predfile.name}')
    st.write("total time(s) : ", total_time)
    st.write("total time(integer) : ", integer)
    st.write("")

    # waveの実データを取得し数値化
    data = wr.readframes(wr.getnframes())
    wr.close()
    X = np.frombuffer(data, dtype='int16')

    # play sound
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=fr*2, data=X)
    st.audio(virtualfile)



    for i in range(num_cut):
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

    return num_cut

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

# youtube_url = st.sidebar.text_input(label="OR YOUTUBE LINK HERE (<4min)👇:")
# if predfile is not None:
#     st.sidebar.write("to use youtube link, please delete all files above.")
# elif youtube_url is None:
#     pass
# else:
#     st.sidebar.write(youtube_url)
#     wr = youtube_to_wav(youtube_url)

if predfile is not None:
    #wr = load_wave_file(predfile)
    wr = wave.open(predfile, 'r')
    n = target_cropper(wr,3)
    #predict_timestamp_and_remove(n)
    components.iframe("http://localhost:8501/test",height=600,scrolling=True)
else:
    pass
