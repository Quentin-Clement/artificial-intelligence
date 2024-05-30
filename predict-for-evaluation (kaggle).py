import librosa
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")
directory = "" #The directory containing the sound files
sample_rate = 8000

def load_audio_file(audio_file_path):
    audio_segment, _ = librosa.load(audio_file_path, sr=sample_rate)
    return audio_segment

def fix_audio_segment_to_10_seconds(audio_segment):
    target_len = 10 * sample_rate
    audio_segment = np.concatenate([audio_segment]*2, axis=0)
    audio_segment = audio_segment[0:target_len]
    
    return audio_segment

def spectrogram(audio_segment):
    image_width = 500
    image_height = 128
    # Compute Mel-scaled spectrogram image
    hl = audio_segment.shape[0] // image_width
    spec = librosa.feature.melspectrogram(audio_segment, n_mels=image_height,hop_length=int(hl))

    # Logarithmic amplitudes
    image = librosa.core.power_to_db(spec)

    # Convert to np matrix
    image_np = np.asmatrix(image)

    # Normalize and scale
    image_np_scaled_temp = (image_np - np.min(image_np))
    
    image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)

    return image_np_scaled[::-1, :image_width]

def to_integer(image_float):
    # range (0,1) -> (0,255)
    image_float_255 = image_float * 255.0
    
    # Convert to uint8 in range [0:255]
    image_int = image_float_255.astype(np.uint8)
    
    return image_int



def processWav(f):
    if os.path.isdir("./out") == False:
        os.mkdir("./out")
    
    audio = fix_audio_segment_to_10_seconds(load_audio_file(f"{directory}/{f}"))
    f = os.path.basename(os.path.basename(f))
    img = to_integer(spectrogram(audio))
    os.system('clear')
    plt.imsave(f"./out/{f[:-4]}.png", img, vmin=0, vmax=255)
    img = Image.open(f"./out/{f[:-4]}.png").convert('L')
    img.save(f"./out/{f[:-4]}.png")
    
    return f"./out/{f[:-4]}.png"
!wget ./model.h5 https://github.com/algosup/2022-Project-Artificial-Intelligence-Group-B/blob/main/model.h5?raw=true
model = load_model("./model.h5?raw=true")
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    path = processWav(filename)
    
    img = Image.open(path)
    img = np.array(img)
    
    img = img/255
    img = img.reshape(1, 128, 500, 1)
    
    prediction = model.predict(img)
    if(prediction[0][0] < 0.5):
        print(f"{path[6:-4]} is French")
    else:
        print(f"{path[6:-4]} is English")