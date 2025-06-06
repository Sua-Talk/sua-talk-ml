import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify,render_template
import soundfile as sf
import tempfile
from dotenv import load_dotenv
from datetime import datetime
import psutil

from utils.ai_insights import generate_ai_insights
from utils.calendar_utils import create_calendar_event

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'classification_model/classifier_model.h5')
PORT = int(os.getenv('PORT', 5000))
MAX_REQUEST_SIZE = os.getenv('MAX_REQUEST_SIZE', '50MB')
CORS_ORIGIN = os.getenv('CORS_ORIGIN', '*')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'info')
ALLOWED_AUDIO_FORMATS = os.getenv('ALLOWED_AUDIO_FORMATS', 'wav,mp3,m4a,flac').split(',')

# Audio processing constants (matching your working code)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MFCC = 13
N_MELS = 128
N_BANDS = 7
FMIN = 100

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max-file-size

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    model_loaded_time = datetime.utcnow()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_loaded_time = None

# Define class labels based on your working code
CLASS_LABELS = ['burping', 'discomfort', 'belly_pain', 'hungry', 'tired']

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

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint for CapRover zero-downtime deployment"""
    try:
        # Check model status
        if model is None:
            return jsonify({
                'status': 'unhealthy',
                'reason': 'model not loaded',
                'timestamp': datetime.utcnow().isoformat()
            }), 500
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 90:
            return jsonify({
                'status': 'unhealthy',
                'reason': 'high memory usage',
                'memory_percent': memory_percent,
                'timestamp': datetime.utcnow().isoformat()
            }), 500
        
        # Check disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            return jsonify({
                'status': 'unhealthy',
                'reason': 'high disk usage',
                'disk_percent': disk_percent,
                'timestamp': datetime.utcnow().isoformat()
            }), 500
        
        # All checks passed
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': True,
            'model_path': MODEL_PATH,
            'model_loaded_time': model_loaded_time.isoformat() if model_loaded_time else None,
            'version': os.getenv('MODEL_VERSION', '1.0.0'),
            'environment': os.getenv('FLASK_ENV', 'development'),
            'uptime_seconds': (datetime.utcnow() - model_loaded_time).total_seconds() if model_loaded_time else 0,
            'system_health': {
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'cpu_count': psutil.cpu_count()
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'reason': f'health check error: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check for Kubernetes/CapRover deployment"""
    if model is None:
        return jsonify({
            'ready': False,
            'reason': 'model not loaded'
        }), 503
    
    return jsonify({
        'ready': True,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available prediction classes"""
    return jsonify({
        'classes': CLASS_LABELS,
        'total_classes': len(CLASS_LABELS)
    })

# # --- Database Helper ---
# def log_cry_event(predicted_label, confidence, timestamp):
#     conn = sqlite3.connect('database/cry_history.db')
#     cursor = conn.cursor()
#     cursor.execute('''CREATE TABLE IF NOT EXISTS cry_events (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         label TEXT, confidence REAL, timestamp TEXT)''')
#     cursor.execute("INSERT INTO cry_events (label, confidence, timestamp) VALUES (?, ?, ?)",
#                    (predicted_label, confidence, timestamp))
#     conn.commit()
#     conn.close()

# def get_cry_history():
#     conn = sqlite3.connect('database/cry_history.db')
#     cursor = conn.cursor()
#     cursor.execute("SELECT label, confidence, timestamp FROM cry_events ORDER BY timestamp DESC LIMIT 100")
#     rows = cursor.fetchall()
#     conn.close()
#     return [{"label": r[0], "confidence": r[1], "timestamp": r[2]} for r in rows]

@app.route('/predict', methods=['POST'])
def predict():
    """Predict infant cry classification"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Check if file is present (supporting both 'audio' and 'file' keys)
    file = None
    if 'audio' in request.files:
        file = request.files['audio']
    elif 'file' in request.files:
        file = request.files['file']
    
    if not file:
        return jsonify({'error': 'No audio file provided'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Check file format
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in ALLOWED_AUDIO_FORMATS:
        return jsonify({
            'error': f'Unsupported audio format. Allowed: {", ".join(ALLOWED_AUDIO_FORMATS)}'
        }), 400
    
    tmp_file_path = None
    try:
        # Save uploaded file temporarily using secure approach
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            tmp_file_path = tmp_file.name
            file.save(tmp_file_path)
            
            # Extract features (with trim_or_pad preprocessing)
            features = extract_features(tmp_file_path)
            
            if features is None:
                return jsonify({'error': 'Feature extraction failed'}), 500
            
            # Make prediction
            prediction = model.predict(features, verbose=0)
            predicted_class_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = CLASS_LABELS[predicted_class_idx]

            import datetime
            # log_cry_event(predicted_class, confidence, datetime.datetime.now().isoformat())
            baby_gender='female'
            baby_age='3 months'
            ai_recommendation = generate_ai_insights(
                label=predicted_class,
                gender=baby_gender,
                age=baby_age
            )
            
    
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'all_predictions': {
                    CLASS_LABELS[i]: float(prediction[0][i]) 
                    for i in range(len(CLASS_LABELS))
                },
                'feature_shape': features.shape,
                'ai-recommendation' : ai_recommendation
            })
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass  # Fail silently on cleanup errors

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size: {MAX_REQUEST_SIZE}'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Enable CORS if specified
@app.after_request
def after_request(response):
    if CORS_ORIGIN != '*':
        response.headers.add('Access-Control-Allow-Origin', CORS_ORIGIN)
    else:
        response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# # --- Insights & Calendar Endpoint ---
# @app.route('/analyze-history', methods=['POST'])
# def analyze_history():
#     history_data = get_cry_history()
#     insights = generate_ai_insights(history_data)
#     # (Optional) create calendar event based on insights['next_feeding'] etc
#     create_calendar_event(insights)
#     return jsonify(insights)

# # --- Simple Dashboard ---
# @app.route('/')
# def dashboard():
#     history = get_cry_history()
#     insights = generate_ai_insights(history)
#     return render_template('dashboard.html', history=history, insights=insights)

if __name__ == '__main__':
    print(f"🚀 Starting SuaTalk ML API")
    print(f"🔧 Port: {PORT}")
    print(f"🔧 Environment: {os.getenv('FLASK_ENV', 'development')}")
    print(f"🔧 Model: {MODEL_PATH}")
    print(f"🔧 CORS Origin: {CORS_ORIGIN}")
    print(f"🔧 Log Level: {LOG_LEVEL}")
    print(f"🔧 Audio Config: SR={SAMPLE_RATE}, MFCC={N_MFCC}, FFT={N_FFT}, HOP={HOP_LENGTH}")
    
    app.run(host='0.0.0.0', port=PORT, debug=(LOG_LEVEL == 'debug'))