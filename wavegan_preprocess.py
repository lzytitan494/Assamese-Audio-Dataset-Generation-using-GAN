import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the paths to the audio files and the output directory
audio_dir = "C:/Users/Yash Vardhan Gautam/OneDrive - iiitnr.edu.in/Documents/Projects/MLA Project/clips"
output_dir = "C:/Users/Yash Vardhan Gautam/OneDrive - iiitnr.edu.in/Documents/Projects/MLA Project/data"

# Loop through each audio file in the directory
for audio_file in os.listdir(audio_dir):
    # Load the audio file
    audio_path = os.path.join(audio_dir, audio_file)
    audio, sr = librosa.load(audio_path, sr=16000)

    # Convert the audio waveform into a spectrogram image
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
    log_S = librosa.power_to_db(S, ref=np.max)
    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    librosa.display.specshow(log_S, cmap='gray_r')
    plt.savefig(os.path.join(output_dir, f"{audio_file}.png"), dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    
print("Done preprocessing audio files!")
