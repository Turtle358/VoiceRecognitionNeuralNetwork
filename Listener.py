import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pyaudio
import wave
import tensorflow as tf
from PIL import Image


def ToWaveform(mp3Path, OutputImagePath='./waveform_image.png'):
    # Load Audio File
    y,sr = librosa.load(mp3Path)
    # Plot Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y,sr=sr)
    plt.axis('off')
    plt.savefig(OutputImagePath,bbox_inches='tight',pad_inches=0,transparent=True)
    plt.close()
def ToSpectogram(mp3Path, OutputImagePath='./spectogram_image.png'):
    # Load File
    y,sr= librosa.load(mp3Path)
    # Calc Spectogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)),ref=np.max)
    # Plot and save
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D,sr=sr,x_axis='time',y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    plt.savefig(OutputImagePath, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
def RecordAudio(output_file='./Recorded.wav', duration=5, samplerate=44100):
    Chunk = 1024
    Format = pyaudio.paInt16
    Channels = 1
    print('Recording audio...')
    p = pyaudio.PyAudio()
    stream = p.open(format=Format, channels=Channels, rate=samplerate, input=True,
                    frames_per_buffer=Chunk)
    frames = []
    num_chunks = int(samplerate * duration / Chunk)  # Calculate the number of chunks needed for the duration
    for i in range(0, num_chunks):
        data = stream.read(Chunk)
        frames.append(data)
    print('Recording Finished')
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(Channels)
    wf.setsampwidth(p.get_sample_size(Format))
    wf.setframerate(samplerate)
    wf.writeframes(b''.join(frames))
    wf.close()
def post_process_predictions(predictions):
    # Assuming predictions is a sequence of characters or tokens
    transcription = ""
    for token in predictions:
        if token == "<end_of_sentence_token>":
            break
        transcription += token

    return transcription
def ConvToText(file='./spectogram_image.png'):
    # Load the model
    model = tf.keras.models.load_model('../../Commands/VoiceRecognition/saved_model')
    image = Image.open(file)
    image_array = np.array(image)
    image_array = image_array.reshape((1,) + image_array.shape)

    prediction = model.predict(image_array)
    #transcription = post_process_predictions(prediction)
    print("Transcription:", prediction)

if __name__ == '__main__':
    RecordAudio()
    ToSpectogram('./Recorded.wav')
    ToWaveform('./Recorded.wav')
    ConvToText()