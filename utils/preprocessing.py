import librosa
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
import pandas as pd


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
    dob = datetime.strptime(date_of_birth_str, "%Y%m%d")
    today = datetime.today()
    delta = today - dob
    # Hitung umur dalam bulan dan hari
    months = delta.days // 30
    days = delta.days % 30
    if months > 0:
        return f"{months} bulan {days} hari"
    else:
        return f"{days} hari"
    
    
def get_baby_history_summary(records, days=30):
    try:
        if not records:
            return "Riwayat tangisan tidak tersedia."
        df = pd.DataFrame(records)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        last_month = datetime.now(timezone.utc) - timedelta(days=days)
        df = df[df['Timestamp'] >= last_month]
        if df.empty or 'prediction' not in df.columns:
            return f"Riwayat tangisan {days} hari terakhir tidak tersedia."

        # Distribusi label prediksi
        pred_counts = df['prediction'].value_counts(normalize=True).sort_values(ascending=False)
        distribusi_label = ', '.join([f"{label}: {pct*100:.1f}%" for label, pct in pred_counts.items()])

        # Pola waktu per label
        pola_jam_per_label = []
        for label in pred_counts.index:
            df_label = df[df['prediction'] == label]
            if not df_label.empty:
                top_hours = df_label['Timestamp'].dt.hour.value_counts().nlargest(2)
                jam_rinci = ', '.join([f"{jam}:00 ({count} kali)" for jam, count in top_hours.items()])
                pola_jam_per_label.append(f"{label}: {jam_rinci}")
        pola_jam_per_label_str = '; '.join(pola_jam_per_label)

        summary = (f"Selama {days} hari terakhir, distribusi penyebab tangisan bayi adalah: {distribusi_label}. "
                   f"Pola jam per label: {pola_jam_per_label_str}.")
        return summary

    except Exception as e:
        return "Riwayat tangisan tidak tersedia.dengan error" , e
    

