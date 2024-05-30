# This should be kept to False
# It will generate audio and spectrogram files for debugging purposes
DEBUG = False

# Imports

from gpiozero import LED
from io import BytesIO
from librosa import feature, load as load_audio, power_to_db
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.io.wavfile import write
from selenium import webdriver
from selenium.webdriver.common.by import By
from sounddevice import InputStream # Takes a long time to import (~10s on our Raspberry)
from tensorflow.keras.models import load_model
from time import sleep, time

# Setting the constants and global values

# Debug settings
LANGS = ["EN", "FR"]
DEBUG_AUDIO_PATH = "/home/groupb/Documents/result.wav"
DEBUG_SPECTROGRAM_PATH = "/home/groupb/Documents/outputs/{}.png"
debug_wav = np.array([])

# Audio settings
CHANNEL_NB = 1
CHUNK_SIZE = 1024  # Record in chunks of 1024 samples
DURATION = 10 # Duration of recording for the prediction
REC_SAMPLE_RATE = 44100

# Spectrogram and model
IMAGE_SIZE = 500, 128
SPEC_SAMPLE_RATE = 8000
MODEL_PATH = "/home/groupb/Documents/model.h5"
img_index = 0

# LEDs configuration (from lowest to highest probability)
LEDS = [
    LED(2), LED(6), LED(5), # French
    LED(4), LED(3), LED(13), # English
    LED(19) # Unknown
]
last_led = None

# LED selection function
MIN_CERTAINTY = 0.5 # Below that percentage, consider it unkown

def choose_led(index, prediction):
    value = prediction[index] # value ∈ [0, 1]
    if value < MIN_CERTAINTY:
        raise ValueError(f"Prediction {value} is lower than the required probability {MIN_CERTAINTY}")

    # Custom LERP from [MIN_CERTAINTY, 1] to [0 .. size[
    size = len(LEDS)
    norm = (1 - value) / (1 - MIN_CERTAINTY) # norm ∈ [0, 1]
    i = size - floor(norm * size) # i ∈ [0 .. size[
    return i

# Selenium configuration
sele_colors = [
    'red', # FR
    'rgb(0, 165, 0)', 'yellowgreen', 'yellow', 'orange', # EN
    'rgb(150, 150, 246)' # Unknown
]
sele_text = [
    'French', # FR
    'Great English', 'Good English', 'Understandable English', 'Bad English', # EN
    'Unknown'
]

# Initiating Selenium

driver = webdriver.Chrome()
driver.get("https://google.com");
f = open("/home/groupb/Documents/serv/index.html", "r")
webElement = driver.find_element(by=By.XPATH, value='html/body');
script = "arguments[0].innerHTML=`"+f.read()+"`"
driver.execute_script(script, webElement);
webElement = driver.find_element(by=By.XPATH, value='//*[@id="Result"]')
webElementtxt = driver.find_element(by=By.XPATH, value='//*[@id="Result"]/h1')

# Tools

def format_time(duration):
    """
    Format time in seconds to string.
    """
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def format_prediction(prediction):
    return "\t".join(
        f"{lang}: {prob}"
        for lang, prob in zip(LANGS, prediction)
    )

def get_prediction_from_audio_file(filepath, model):
    audio_buffer, _ = load_audio(filepath, sr=REC_SAMPLE_RATE)
    spec = get_spectrogram(audio_buffer, SPEC_SAMPLE_RATE, img_size=IMAGE_SIZE)
    img = spectrogram_to_grayscale(spec) / 255 # Converting because colors are unnecessary
    prediction = model.predict(img.reshape(1, 128, 500, 1))[0]
    print(filepath, ":")
    activate_leds(prediction)

def cleanup(audio_path="", image_path=""):
    """
    Clean up temporary files and connections.
    """
    if DEBUG and os.path.exists(audio_path):
        os.remove(audio_path)
    if DEBUG and os.path.exists(image_path):
        os.remove(image_path)
    for led in LEDS:
        led.close()

# Functions

def expand_audio_buffer(indata, frames, time, status):
    """
    Process audio_buffer data into a spectrogram.

    Callback for sd.InputStream.
    """
    global audio_buffer, debug_wav
    if DEBUG:
        debug_wav = np.append(debug_wav, indata.copy())
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata.reshape(-1,)

def get_spectrogram(audio, sample_rate, img_size):
    """
    Get image spectrogram from audio.

    Callback for the sounddevice library.
    """
    #audio = np.where(np.isnan(audio), 0, audio)
    width, height = img_size
    try:
        spectrogram = feature.melspectrogram(y=audio, sr=sample_rate, hop_length=len(audio) // width, n_mels=height)
        matrix = power_to_db(spectrogram)[::-1, :width]
        return matrix

    except RuntimeWarning:
        pass

def spectrogram_to_grayscale(spectrogram):
    """
    Process spectrogram into a gray-scale array.
    """
    global img_index  # Debug
    with BytesIO() as buffer:
        plt.imsave(buffer, spectrogram, cmap="gray")
        buffer.seek(0)
        img = Image.open(buffer).convert("L")
    img = np.array(img, dtype=np.float32)
    return img

def get_prediction(audio, model):
    spec = get_spectrogram(audio, SPEC_SAMPLE_RATE, img_size=IMAGE_SIZE)
    spec = spectrogram_to_grayscale(spec) # Converting because colors are unnecessary
    spec = spec.reshape(1, *spec.shape[:2], 1) / 255 # Reshape and normalize

    predictions = model.predict(spec)
    # The prediction is a list of lists of probabilities.
    # The matching labels are [EN, FR] by default, can be changed in LANGS

    return predictions[0]

def activate_leds(prediction):
    global last_led

    val = np.max(prediction).round(decimals=2)
    highest_probability_index = 0
    arr_en=False

    if np.argmax(prediction) == 0:
        arr_en=True
        highest_probability_index += 1
    print(LANGS[np.argmax(prediction)] + "  :  " + str(val))

    if val > .90:
        highest_probability_index += 0
    elif val > .80:
        highest_probability_index += 1
    elif val > .70:
        highest_probability_index += 2
    elif val > .60:
        highest_probability_index += 3
    else:
        highest_probability_index = -1

    if last_led is not None:
        last_led.off()
    if not arr_en and val < .70:
        highest_probability_index = -1
    elif not arr_en and val > .70:
        highest_probability_index = 0

    script = "arguments[0].style.backgroundColor = '" + sele_colors[highest_probability_index] + "'"
    driver.execute_script(script, webElement)
    script = "arguments[0].innerHTML = '" + sele_text[highest_probability_index] + "'"
    driver.execute_script(script, webElementtxt)
    
    last_led = LEDS[highest_probability_index]
    last_led.on()
    return last_led

# Main

def main():
    """
    Listen to the microphone and process the audio data into spectrograms.
    """
    global audio_buffer
    
    model = load_model(MODEL_PATH, compile=False)

    audio_buffer = np.empty(
        (int(REC_SAMPLE_RATE * DURATION),),
        dtype=np.float32
    )

    # Create an audio stream and record to the audio buffer.
    with InputStream(blocksize=CHUNK_SIZE, samplerate=REC_SAMPLE_RATE, channels=CHANNEL_NB, callback=expand_audio_buffer):
        print("Start")

        try:
            while True: # Use Ctrl+C to trigger a KeyboardInterrupt and exit

                if len(audio_buffer) < CHUNK_SIZE:
                    continue

                prediction = get_prediction(audio_buffer, model)
                activate_leds(prediction)

                if DEBUG:
                    print(format_prediction(prediction))

        except KeyboardInterrupt:
            pass

    if DEBUG:
        global debug_wav
        debug_wav = np.round(debug_wav / np.max(debug_wav) * 32767).astype(np.int16)
        write(DEBUG_AUDIO_PATH, REC_SAMPLE_RATE, debug_wav)

    print("Done")

if __name__ == "__main__":
	main()