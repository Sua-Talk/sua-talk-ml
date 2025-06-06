from openai import OpenAI

def generate_ai_insights(label, gender, age):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "user",
                "content": f"A {age}-month-old {gender} baby is crying because of {label}. Please provide brief action recommendations to help new parents. Make it in one paragraph, quick treatment"
            }
        ],
        max_tokens=500
    )

    return (completion.choices[0].message.content)
