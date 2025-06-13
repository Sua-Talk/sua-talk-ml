import os
import json
import tempfile
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.oauth2 import service_account


load_dotenv()

#Load Variabel Lingkungan
GOOGLE_CLOUD_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_REGION = os.getenv('GOOGLE_CLOUD_REGION')
VERTEX_AI_MODEL_ENDPOINT = os.getenv('VERTEX_AI_MODEL_ENDPOINT')


def create_credentials_from_env():
    """Create Google credentials object from environment variable JSON"""
    credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    
    if not credentials_json:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is not set")
    
    if not credentials_json.strip():
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable is empty")
    
    try:
        # Parse the JSON string
        credentials_info = json.loads(credentials_json)
        
        # Fix private key format - convert escaped newlines to actual newlines
        if 'private_key' in credentials_info:
            private_key = credentials_info['private_key']
            # Replace escaped newlines with actual newlines
            private_key = private_key.replace('\\n', '\n')
            credentials_info['private_key'] = private_key
        
        # Create credentials object from the parsed JSON
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        return credentials
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
    except Exception as e:
        print(f"❌ Credentials creation error: {e}")
        raise ValueError(f"Failed to create credentials: {e}")


def generate(label, age, history_summary):
    """ Fungsi untuk menggenerate rekomendasi dari vertex AI.
    """
    try:
        # Create credentials from environment variable
        credentials = create_credentials_from_env()
        
        client = genai.Client(
          vertexai=True,
          project=GOOGLE_CLOUD_PROJECT_ID,
          location=GOOGLE_CLOUD_REGION,
          credentials=credentials
      )
        prompt_text = (
            f"Anda adalah asisten AI yang ahli dalam memberikan rekomendasi perawatan bayi berdasarkan penyebab tangisan. "
            f"Bayi menangis adalah hal yang normal, dan setiap tangisan memiliki arti. Tujuan Anda adalah memberikan rekomendasi "
            f"yang lembut dan efektif kepada orang tua baru. "
            f"Penyebab tangisan bayi saat ini terklasifikasi sebagai: {label}. "
            f"Usia bayi saat ini adalah: {age}. "
            f"Berdasarkan riwayat tangisan sebelumnya: {history_summary}. "
            f"Berdasarkan informasi di atas, berikan beberapa rekomendasi treatment yang bisa dilakukan orang tua baru untuk "
            f"mengatasi tangisan ini. Prioritaskan rekomendasi yang paling umum dan relevan dengan pola yang teridentifikasi. "
            f"Sajikan rekomendasi dan penjelasannya dalam bentuk satu paragraf singkat."
        )
        msg1_text1 = types.Part.from_text(text=prompt_text)
        model = VERTEX_AI_MODEL_ENDPOINT or f"projects/{GOOGLE_CLOUD_PROJECT_ID}/locations/{GOOGLE_CLOUD_REGION}/endpoints/598359725393838080"
        contents = [
            types.Content(
                role="user",
                parts=[msg1_text1]
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
        )
        output_chunks = []
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            output_chunks.append(chunk.text)
        return ''.join(output_chunks)
    
    except Exception as e:
        print(f"Error in generate function: {e}")
        raise e