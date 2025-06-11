import librosa
import numpy as np
import datetime
import requests

# Audio processing constants (matching your working code)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MFCC = 13
N_MELS = 128
N_BANDS = 7
FMIN = 100

def trim_or_pad(audio, sr, target_duration=6.0):
    target_length = int(sr * target_duration)
    if len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    elif len(audio) < target_length:
        pad_length = target_length - len(audio)
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    return audio

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = trim_or_pad(y, sr=SAMPLE_RATE)
        
        mfcc = np.mean(librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window='hann'
        ).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, window='hann', n_mels=N_MELS
        ).T, axis=0)

        stft = np.abs(librosa.stft(y))

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T, axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(
            S=stft, y=y, sr=sr, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            n_bands=N_BANDS, fmin=FMIN
        ).T, axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)

        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features.reshape(1, -1)

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def calculate_baby_age(date_of_birth_str):
    # Format date_of_birth_str: "YYYY-MM-DD"
    dob = datetime.strptime(date_of_birth_str, "%Y-%m-%d")
    today = datetime.today()
    delta = today - dob
    # Hitung umur dalam bulan dan hari
    months = delta.days // 30
    days = delta.days % 30
    if months > 0:
        return f"{months} bulan {days} hari"
    else:
        return f"{days} hari"
    
def get_baby_profile(baby_id):
    url = f"https://api.suatalk.site/babies/{baby_id}"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("success") and data.get("data"):
            return data["data"]
        return None
    except Exception:
        return None