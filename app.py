import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
import soundfile as sf
import tempfile
from dotenv import load_dotenv
from datetime import datetime
import psutil

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

# Define class labels based on the notebook
CLASS_LABELS = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']

def extract_features(audio_file):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Extract MFCC features (13 coefficients as per the notebook)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Take mean across time dimension
        mfccs_mean = np.mean(mfccs, axis=1)
        
        return mfccs_mean.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error processing audio file: {str(e)}")

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

@app.route('/predict', methods=['POST'])
def predict():
    """Predict infant cry classification"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    # Check if file is present
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Check file format
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in ALLOWED_AUDIO_FORMATS:
        return jsonify({
            'error': f'Unsupported audio format. Allowed: {", ".join(ALLOWED_AUDIO_FORMATS)}'
        }), 400
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            file.save(tmp_file.name)
            
            # Extract features
            features = extract_features(tmp_file.name)
            
            # Make prediction
            prediction = model.predict(features, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = CLASS_LABELS[predicted_class_idx]
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence,
                'all_predictions': {
                    CLASS_LABELS[i]: float(prediction[0][i]) 
                    for i in range(len(CLASS_LABELS))
                }
            })
            
    except Exception as e:
        # Clean up temporary file if it exists
        try:
            os.unlink(tmp_file.name)
        except:
            pass
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    print(f"🚀 Starting SuaTalk ML API")
    print(f"🔧 Port: {PORT}")
    print(f"🔧 Environment: {os.getenv('NODE_ENV', 'development')}")
    print(f"🔧 Model: {MODEL_PATH}")
    print(f"🔧 CORS Origin: {CORS_ORIGIN}")
    print(f"🔧 Log Level: {LOG_LEVEL}")
    
    app.run(host='0.0.0.0', port=PORT, debug=(LOG_LEVEL == 'debug'))
