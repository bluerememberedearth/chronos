import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found.")
else:
    genai.configure(api_key=api_key)
    
    # List of models from previous programmatic run
    models_to_test = [
        "gemini-1.0-pro-vision-latest",
        "gemini-1.5-pro",
        "gemini-1.5-pro-002",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-8b-001",
        "gemini-1.5-flash-8b-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-12-2025",
        "gemini-2.5-flash",
        "gemini-2.5-flash-8b",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-image-preview",
        "nano-banana-pro-preview",
        "gemini-robotics-er-1.5-preview",
        "gemini-2.5-computer-use-preview-10-2025",
        "deep-research-pro-preview-12-2025"
    ]
    
    print(f"{'Model':<45} | {'State'}")
    print("-" * 60)
    
    results = []
    for m_name in models_to_test:
        try:
            model = genai.GenerativeModel(m_name)
            # Very small request to minimize quota usage
            response = model.generate_content("ping", generation_config={"max_output_tokens": 5})
            print(f"{m_name:<45} | SUCCESS")
            results.append((m_name, "SUCCESS"))
        except Exception as e:
            err_msg = str(e).split('\n')[0]
            print(f"{m_name:<45} | FAILED: {err_msg[:30]}...")
            results.append((m_name, f"FAILED: {err_msg}"))
        
        # Minor sleep to avoid hitting RPM limits during the "ping" test
        time.sleep(2)

    print("\nSummary:")
    success_count = sum(1 for _, res in results if res == "SUCCESS")
    print(f"Total: {len(models_to_test)}, Success: {success_count}, Failed: {len(models_to_test) - success_count}")
