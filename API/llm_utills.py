import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv(""))

def get_disease_description(disease_name):
    prompt = f"""
    You are an agricultural expert.
    Explain the potato leaf disease: {disease_name}.

    Provide:
    1. Short description
    2. Symptoms
    3. Causes
    4. Prevention methods

    Keep it concise and farmer-friendly.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
