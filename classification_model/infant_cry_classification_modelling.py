# -*- coding: utf-8 -*-
"""Infant_Cry_Classification_Modelling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/Sua-Talk/sua-talk-ml/blob/main/classification_model/Infant_Cry_Classification_Modelling.ipynb

## Load Data
"""

# File and system operations
import os
import random

# Audio processing
import librosa
import librosa.display
import soundfile as sf

# Numerical and data handling
import numpy as np
import pandas as pd

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Deep learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

os.environ["PYTHONHASHSEED"] = str(42)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

np.random.seed(42)
random.seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


SAMPLE_RATE = 16000
def load_audio(file_path, sr=SAMPLE_RATE):
    """Loads an audio file."""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

def save_audio(audio, output_file_path, sr = SAMPLE_RATE):
    """Saves an audio file."""
    sf.write(output_file_path, audio, sr)

def time_shift(audio,sr, shift_range=(-500, 500)):
    """Shifts the audio signal in time."""
    shift = np.random.randint(shift_range[0], shift_range[1])
    return np.roll(audio, shift)

def time_stretch(audio,sr, rate_range=(0.8, 1.2)):
    """Stretches or compresses the audio signal without changing pitch."""
    rate = np.random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr=SAMPLE_RATE, semitones=4):
    """Changes the pitch of the audio signal."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)

def add_white_noise(audio,sr, mean=0, variance=0.000025, power=0.005):
    """Adds white noise to the audio signal."""
    noise = np.random.normal(mean, np.sqrt(variance), len(audio))
    audio_amplitude = np.max(np.abs(audio))
    scaled_noise = noise * audio_amplitude * power
    return audio + scaled_noise

def audio_slice(audio, sr = SAMPLE_RATE, duration=1):
    slice_length = int(duration * sr)
    if len(audio) <= slice_length:
        return librosa.util.fix_length(audio, size=slice_length)
    start = random.randint(0, len(audio) - slice_length)
    return audio[start:start + slice_length]

AUGMENTATION_FUNCTIONS = [
    (time_shift, "shift"),
    (time_stretch, "stretch"),
    (pitch_shift, "pitch"),
    (add_white_noise, "noise"),
    (audio_slice, "slice")
]

"""## Feature Extraction"""

n_mfcc = 13
n_fft = 1024
hop_length = 10*16
win_length = 25*16
window = 'hann'
n_chroma=12
n_mels=128
n_bands=7
fmin=100
bins_per_ocatve=12

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',n_mels=n_mels).T,axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr,n_fft=n_fft,
                                                      hop_length=hop_length, win_length=win_length,
                                                      n_bands=n_bands, fmin=fmin).T,axis=0)
        tonnetz =np.mean(librosa.feature.tonnetz(y=y, sr=sr).T,axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        # print(shape(features))
        return features
    except:
        print("Error: Exception occurred in feature extraction")
        return None

path = 'donateacry-corpus/donateacry_corpus_cleaned_and_updated_data'
features = []
labels = []

for label in os.listdir(path):
    label_path = os.path.join(path, label)
    if os.path.isdir(label_path):
        print(f"{label} data is loading.....")
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
        print(f"{label} data loaded....")

"""## Modelling"""

features = np.array(features)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, num_classes=5)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=42, stratify=labels)

input_shape=(X_train.shape[1],)
num_classes=5

model_ANN = Sequential([
    InputLayer(shape=input_shape),

    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    # ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     patience=5,
    #     min_lr=1e-6,
    #     verbose=1
    # )
]


model_ANN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_ANN.summary()

history_ANN=model_ANN.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.1)

y_pred=np.argmax(model_ANN.predict(X_test), axis=1)
y_true=np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=le.classes_))

plt.plot(history_ANN.history['loss'])
plt.plot(history_ANN.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

y_test_bin = label_binarize(y_true, classes=np.arange(num_classes))
y_pred_prob = model_ANN.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label='ROC of {0} (area = {1:0.2f})'
                                   ''.format(le.classes_[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

model_ANN.save('classifier_model.h5')
model_ANN.save('classifier_model.keras')