from google import genai
from google.genai import types

def generate(label, age, history_summary):
  client = genai.Client(
      vertexai=True,
      project="330163298455",
      location="us-central1",
  )
  prompt_text=""f"Anda adalah asisten AI yang ahli dalam memberikan rekomendasi perawatan bayi berdasarkan penyebab tangisan. Bayi menangis adalah hal yang normal, dan setiap tangisan memiliki arti. Tujuan Anda adalah memberikan rekomendasi yang lembut dan efektif kepada orang tua baru.Penyebab tangisan bayi saat ini terklasifikasi sebagai: {label}.Usia bayi saat ini adalah: {age}.Berdasarkan riwayat tangisan sebelumnya:{history_summary}.Berdasarkan informasi di atas, berikan beberapa rekomendasi treatment yang bisa dilakukan orang tua baru untuk mengatasi tangisan ini. Prioritaskan rekomendasi yang paling umum dan relevan dengan pola yang teridentifikasi. Sajikan rekomendasi dan penjelasannya dalam bentuk satu paragraf singkat"""
  msg1_text1 = types.Part.from_text(text=prompt_text)
  model = "projects/330163298455/locations/us-central1/endpoints/598359725393838080"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )
  output_chunks = []
  for chunk in client.models.generate_content_stream(
      model=model,
      contents=contents,
      config=generate_content_config,):
      output_chunks.append(chunk.text)

  return ''.join(output_chunks)