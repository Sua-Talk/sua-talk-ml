# SuaTalk ML Service

Machine Learning service for infant cry classification using TensorFlow/Keras.

## Features

- **Cry Classification**: Classifies infant cries into 5 categories:

  - `belly_pain` - Stomach discomfort
  - `burping` - Need to burp
  - `discomfort` - General discomfort
  - `hungry` - Hunger
  - `tired` - Fatigue/sleepiness

- **REST API**: Simple HTTP endpoints for predictions
- **Health Monitoring**: Built-in health checks
- **Docker Support**: Ready for containerized deployment

## Quick Start

### Local Development

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**:

   ```bash
   python app.py
   ```

3. **Test the service**:
   ```bash
   python test_api.py
   ```

### Docker Deployment

1. **Build the image**:

   ```bash
   docker build -t suatalk-ml .
   ```

2. **Run the container**:
   ```bash
   docker run -p 5000:5000 suatalk-ml
   ```

### CapRover Deployment

1. **Initialize CapRover app**:

   ```bash
   caprover deploy
   ```

2. **Or deploy via CLI**:
   ```bash
   # Zip the project and upload
   tar --exclude='.git' --exclude='venv' --exclude='__pycache__' -czf suatalk-ml.tar.gz .
   ```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Get Classes

```bash
GET /classes
```

Response:

```json
{
  "classes": ["belly_pain", "burping", "discomfort", "hungry", "tired"],
  "total_classes": 5
}
```

### Predict

```bash
POST /predict
Content-Type: multipart/form-data
```

Body:

- `audio`: Audio file (WAV, MP3, etc.)

Response:

```json
{
  "predicted_class": "hungry",
  "confidence": 0.85,
  "all_probabilities": {
    "belly_pain": 0.1,
    "burping": 0.05,
    "discomfort": 0.0,
    "hungry": 0.85,
    "tired": 0.0
  }
}
```

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:5000/health

# Get classes
curl http://localhost:5000/classes

# Predict (replace with actual audio file)
curl -X POST -F "audio=@sample.wav" http://localhost:5000/predict
```

### Test with Python script

```bash
# Basic tests
python test_api.py

# Test with audio file
python test_api.py path/to/audio.wav
```

## Model Information

- **Framework**: TensorFlow/Keras
- **Input**: Audio files (converted to MFCC features)
- **Output**: Classification probabilities for 5 cry types
- **Model File**: `classification_model/classifier_model.h5`

## Environment Variables

- `PORT`: API port (default: 5000)
- `ML_API_URL`: Base URL for testing (default: http://localhost:5000)

## Dependencies

- Python 3.12+
- TensorFlow 2.16+
- librosa 0.11+
- Flask 3.1+
- NumPy, SciPy, scikit-learn
- Gunicorn (for production)

## Development

### Project Structure

```
sua-talk-ml/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── captain-definition    # CapRover deployment config
├── test_api.py          # API testing script
├── README.md            # This file
└── classification_model/
    └── classifier_model.h5  # Trained ML model
```

### Adding New Features

1. Modify `app.py` for new endpoints
2. Update `requirements.txt` for new dependencies
3. Add tests in `test_api.py`
4. Update this README

## Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `classifier_model.h5` exists in `classification_model/`
2. **Audio processing errors**: Check if audio file format is supported
3. **Missing dependencies**: Run `pip install -r requirements.txt`

### Logs

Check application logs for detailed error information:

```bash
# Docker logs
docker logs <container_id>

# Local development
python app.py  # Will show console output
```
# Test CapRover Native CI/CD Deployment 🚀
