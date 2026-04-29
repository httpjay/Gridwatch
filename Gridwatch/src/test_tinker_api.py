import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
api_key = os.getenv("TINKER_API_KEY")

models_to_test = [
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-V3.1-Base",
    "moonshotai/Kimi-K2-Thinking",
]

print("API key found:", bool(api_key))

client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key,
)

for model_name in models_to_test:
    print(f"\nTesting model: {model_name}")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one sentence."},
            ],
        )
        print("Success")
        print(response.choices[0].message.content)
    except Exception as e:
        print("Failed")
        print(str(e))