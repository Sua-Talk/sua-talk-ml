import os
import tempfile

from dotenv import load_dotenv

from google import genai
from google.genai import types


load_dotenv()

GOOGLE_CLOUD_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_REGION = os.getenv('GOOGLE_CLOUD_REGION')
VERTEX_AI_MODEL_ENDPOINT = os.getenv('VERTEX_AI_MODEL_ENDPOINT')


# def _setup_google_credentials():
#     service_account_info_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
#     if service_account_info_json:
#         try:
#             with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_key_file:
#                 temp_key_file.write(service_account_info_json)
#                 temp_key_file_path = temp_key_file.name
#             os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_key_file_path
#             print(f"✅ Kredensial Akun Layanan dimuat dari variabel lingkungan ke file sementara: {temp_key_file_path}")
#         except Exception as e:
#             print(f"❌ Gagal memuat kredensial Akun Layanan dari variabel lingkungan: {e}")
#     else:
#         print("Peringatan: Variabel lingkungan GOOGLE_APPLICATION_CREDENTIALS_JSON tidak ditemukan. Mengandalkan kredensial default lingkungan.")
        
# _setup_google_credentials()

def setup_vertex():
    import os
    import vertexai

    if os.environ.get("ENV") != "production":
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service_account.json"

    vertexai.init(project=GOOGLE_CLOUD_PROJECT_ID, location=GOOGLE_CLOUD_REGION)

setup_vertex()

def generate(label, age, history_summary):
    client = genai.Client(
      vertexai=True,
      project=GOOGLE_CLOUD_PROJECT_ID,
      location=GOOGLE_CLOUD_REGION,
  )
    """
    Generate a care recommendation using Vertex AI, automatically fetching history for the baby_id.
    """
    
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
    model = "projects/330163298455/locations/us-central1/endpoints/598359725393838080"
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

print(generate('sakit perut', ' 2 months', 'no ratings'))