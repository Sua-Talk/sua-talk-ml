from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os
import shutil

# Konstanta sesuai dengan training
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MFCC = 13
N_MELS = 128
N_BANDS = 7
FMIN = 100

# Path model
MODEL_PATH = os.path.join("classification_model", "classifier_model.h5")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

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
        

# Fungsi ekstraksi fitur audio (sama persis dengan di notebook)
def extract_features(file_path):
    try:
        
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y= trim_or_pad(y, sr=SAMPLE_RATE)
                
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
        print("Error:", e)
        return None

# Inisialisasi Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Simpan file ke temporary path manual (bukan open + locked)
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)
    print(f"📦 Saved to temp: {temp_path}")

    try:
        features = extract_features(temp_path)

        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        prediction = model.predict(features)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        label_map = {
            0: "burping",
            1: "discomfort",
            2: "belly_pain",
            3: "hungry",
            4: "tired"
        }

        result = label_map.get(predicted_class, "unknown")
        print(f"✅ Prediction success: {result} ({confidence})")

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })
    except Exception as e:
        print("❌ Exception:", str(e))
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
