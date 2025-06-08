from google import genai
from google.genai import types

def generate(label, age, gender):
  client = genai.Client(
      vertexai=True,
      project="330163298455",
      location="us-central1",
  )

  msg1_text1 = types.Part.from_text(text=""f"A {age}-old {gender} baby is crying because of {label}. Please provide brief action recommendations to help new parents. Make it in one paragraph, quick treatment""")

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