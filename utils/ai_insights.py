import os
import tempfile
from datetime import datetime, timedelta
import random

import pandas as pd
from dotenv import load_dotenv

from google import genai
from google.genai import types

import requests

load_dotenv()

GOOGLE_CLOUD_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_REGION = os.getenv('GOOGLE_CLOUD_REGION')
VERTEX_AI_MODEL_ENDPOINT = os.getenv('VERTEX_AI_MODEL_ENDPOINT')

def _setup_google_credentials():
    service_account_info_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
    if service_account_info_json:
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_key_file:
                temp_key_file.write(service_account_info_json)
                temp_key_file_path = temp_key_file.name
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_key_file_path
            print(f"✅ Kredensial Akun Layanan dimuat dari variabel lingkungan ke file sementara: {temp_key_file_path}")
        except Exception as e:
            print(f"❌ Gagal memuat kredensial Akun Layanan dari variabel lingkungan: {e}")
    else:
        print("Peringatan: Variabel lingkungan GOOGLE_APPLICATION_CREDENTIALS_JSON tidak ditemukan. Mengandalkan kredensial default lingkungan.")
        
_setup_google_credentials()

# --- DATA DUMMY ---
def generate_dummy_history(num_entries=60, days=30):
    labels = ['lapar', 'lelah', 'tidak nyaman', 'kembung', 'sakit perut']
    now = datetime.now()
    dummy_data = []
    for i in range(num_entries):
        label = random.choices(labels, weights=[0.3, 0.2, 0.3, 0.15, 0.05])[0]
        # acak waktu dalam 30 hari terakhir
        timestamp = now - timedelta(days=random.randint(0, days-1), hours=random.randint(0, 23))
        dummy_data.append({
            "prediction": label,
            "Timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        })
    return dummy_data

def get_baby_history_summary_dummy(records, days=30):
    """
    Versi testing: Merangkum distribusi prediksi dan pola waktu dari data dummy.
    """
    if not records:
        return "Riwayat tangisan tidak tersedia."
    df = pd.DataFrame(records)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    last_month = datetime.now() - timedelta(days=days)
    df = df[df['Timestamp'] >= last_month]
    if df.empty or 'prediction' not in df:
        return "Riwayat tangisan tidak tersedia."
    # Hitung distribusi semua label prediksi
    pred_counts = df['prediction'].value_counts(normalize=True).sort_values(ascending=False)
    distribusi_label = ', '.join([f"{label}: {pct*100:.1f}%" for label, pct in pred_counts.items()])
    # Pola waktu: jam-jam yang sering (top 3 jam)
    top_hours = df['Timestamp'].dt.hour.value_counts().nlargest(3)
    pola_jam = ', '.join([f"{jam}:00 ({count} kali)" for jam, count in top_hours.items()])
    summary = (f"Selama {days} hari terakhir, distribusi penyebab tangisan bayi adalah sebagai berikut: {distribusi_label}. "
               f"Tangisan paling sering terjadi pada jam: {pola_jam}.")
    return summary

def get_baby_history_summary(baby_id, days=30):
    """
    Mengambil dan merangkum distribusi hasil prediksi serta pola waktu per label tangisan bayi selama periode tertentu.
    Endpoint histori bayi: /babies/{baby_id}
    """
    url = f"https://api.suatalk.site/ml/history/{baby_id}?page=1"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        records = data.get('data', [])
        if not records:
            return "Riwayat tangisan tidak tersedia."
        df = pd.DataFrame(records)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        last_month = datetime.now() - timedelta(days=days)
        df = df[df['Timestamp'] >= last_month]
        if df.empty or 'prediction' not in df:
            return "Riwayat tangisan tidak tersedia."

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
        return "Riwayat tangisan tidak tersedia."

# Testing Dummy Data
# def generate(label, age):
#     """
#     Generate a care recommendation using Vertex AI, automatically fetching history for the baby_id.
#     """
#     dummy_records = generate_dummy_history(num_entries=100, days=30)
#     summary = get_baby_history_summary_dummy(dummy_records, days=30)
    
#     client = genai.Client(
#         vertexai=True,
#         project="330163298455",
#         location="us-central1",
#     )
#     prompt_text = (
#         f"Anda adalah asisten AI yang ahli dalam memberikan rekomendasi perawatan bayi berdasarkan penyebab tangisan. "
#         f"Bayi menangis adalah hal yang normal, dan setiap tangisan memiliki arti. Tujuan Anda adalah memberikan rekomendasi "
#         f"yang lembut dan efektif kepada orang tua baru. "
#         f"Penyebab tangisan bayi saat ini terklasifikasi sebagai: {label}. "
#         f"Usia bayi saat ini adalah: {age}. "
#         f"Berdasarkan riwayat tangisan sebelumnya: {summary}. "
#         f"Berdasarkan informasi di atas, berikan beberapa rekomendasi treatment yang bisa dilakukan orang tua baru untuk "
#         f"mengatasi tangisan ini. Prioritaskan rekomendasi yang paling umum dan relevan dengan pola yang teridentifikasi. "
#         f"Sajikan rekomendasi dan penjelasannya dalam bentuk satu paragraf singkat."
#     )
#     msg1_text1 = types.Part.from_text(text=prompt_text)
#     model = "projects/330163298455/locations/us-central1/endpoints/598359725393838080"
#     contents = [
#         types.Content(
#             role="user",
#             parts=[msg1_text1]
#         ),
#     ]
#     generate_content_config = types.GenerateContentConfig(
#         temperature=1,
#         top_p=0.95,
#         max_output_tokens=8192,
#         safety_settings=[
#             types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
#             types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
#             types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
#             types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
#         ],
#     )
#     output_chunks = []
#     for chunk in client.models.generate_content_stream(
#         model=model,
#         contents=contents,
#         config=generate_content_config,
#     ):
#         output_chunks.append(chunk.text)
#     return ''.join(output_chunks)

def generate(label, age, history_summary, baby_id):
    history_summary = get_baby_history_summary(baby_id, days=30)

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