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

# hide "created by streamlit"
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# youtube loader
# ydl_opts = {
#     'format': 'bestaudio/best',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'wav',
#     }],
# }

# def youtube_to_wav(youtube_url):
#     with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([youtube_url])
#         stream = ffmpeg.input('output.m4a')
#         stream = ffmpeg.output(stream, 'output.wav')
#         wr = wave.open('output.wav', 'r')
#     return wr

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


def audio_info(wr):
    #waveファイルが持つ性質を取得
    fr = wr.getframerate()#サンプリング周波数
    fn = wr.getnframes()#フレームの総数
    total_time = 1.0 * fn / fr
    integer = math.floor(total_time)

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
    return

# youtube_url = st.sidebar.text_input(label="OR YOUTUBE LINK HERE (<4min)👇:")
# if predfile is not None:
#     st.sidebar.write("to use youtube link, please delete all files above.")
# elif youtube_url is None:
#     pass
# else:
#     st.sidebar.write(youtube_url)
#     wr = youtube_to_wav(youtube_url)

import requests

predfile = st.sidebar.file_uploader("Upload file", type=['wav'])

if predfile is not None:
    # with open(os.path.join("pages",predfile.name),"wb") as f:
    #     f.write(predfile.getbuffer())
    # wr = wave.open('pages/'+predfile.name, 'r')
    # audio_info(wr)
    # components.iframe("http://localhost:8501/test",height=400,scrolling=True)
    # #st.sidebar.download_button('Download CSV', str(result), 'text/csv')

    url = 'https://ba75-124-219-136-119.ngrok.io/item/'
    with open(predfile, 'rb') as fobj:
        print(fobj)
        test_response = requests.post(url, data=fobj)
        print(test_response)
else:
    pass
