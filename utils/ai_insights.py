import openai
import json
from openrouter import OpenRouter
import os

# OpenRouter API initialization
openrouter = OpenRouter(api_key=os.getenv('OPENROUTER_API_KEY'))

def generate_ai_insights(history):
    prompt = (
        "Berikut adalah data histori tangisan bayi (label, confidence, timestamp):\n"
        f"{history}\n"
        "Buat insight singkat dalam format JSON dengan fields:\n"
        "- next_feeding\n- sleep_time\n- pattern_analysis\n- recommendations\n"
    )

    response = openrouter.predict(
        model="GPT-J", prompt=prompt, max_tokens=250
    )
    
    return {
        "next_feeding": response['choices'][0]['text'],  # Example, adjust accordingly
        "sleep_time": "9:00 AM",
        "pattern_analysis": "Feeding occurs every 3 hours.",
        "recommendations": "Try changing the milk formula."
    }