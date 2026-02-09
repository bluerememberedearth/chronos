import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)
    print(f"{'Model Name':<40} | {'Supported Methods'}")
    print("-" * 80)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"{m.name:<40} | {', '.join(m.supported_generation_methods)}")
    except Exception as e:
        print(f"Error listing models: {e}")
