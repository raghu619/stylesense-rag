import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
key = os.getenv("OPENAI_API_KEY")
print(f"Key loaded, begins {key[:8]}" if key else "NO KEY FOUND")

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["a linen shirt for a hot beach day"],
)
vector = response.data[0].embedding
print(f"Embedding works. Dimensions: {len(vector)}")