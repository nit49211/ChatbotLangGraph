import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Example usage
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("Explain me the concept of LangCHain and LangGraph in simple terms.")
print("Response from Gemini:")
print(response.text)
