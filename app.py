import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import tempfile
from dotenv import load_dotenv
from datetime import datetime
import psutil
from utils.ai_insights import generate
from utils.preprocessing import *
import requests

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

# Define class labels based on your working code
CLASS_LABELS = ['sakit perut', 'kembung', 'tidak nyaman', 'lapar', 'lelah']



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
    
def predict():
    """Predict infant cry classification"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    baby_id = request.form.get('baby_id') or request.json.get('baby_id') if request.is_json else None
    if not baby_id:
        return jsonify({'error': 'baby_id wajib disertakan'}), 400

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
            
            prediction = model.predict(features, verbose=0)
            predicted_class_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = CLASS_LABELS[predicted_class_idx]

            baby_profile = get_baby_profile(baby_id)
            if not baby_profile:
                return jsonify({'error': 'baby profile not found'}), 404
            age = calculate_baby_age(baby_profile["dateOfBirth"])
            
            # Dummy data for testing
            # age = "3 bulan 10 hari"

            #  label, age, baby_id
            ai_recommendation = generate(
                label=predicted_class,
                age=age,
                baby_id=baby_id,
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

if __name__ == '__main__':
    print(f"🚀 Starting SuaTalk ML API")
    print(f"🔧 Port: {PORT}")
    print(f"🔧 Environment: {os.getenv('FLASK_ENV', 'development')}")
    print(f"🔧 Model: {MODEL_PATH}")
    print(f"🔧 CORS Origin: {CORS_ORIGIN}")
    print(f"🔧 Log Level: {LOG_LEVEL}")
    print(f"🔧 Audio Config: SR={SAMPLE_RATE}, MFCC={N_MFCC}, FFT={N_FFT}, HOP={HOP_LENGTH}")
    
    app.run(host='0.0.0.0', port=PORT, debug=(LOG_LEVEL == 'debug'))