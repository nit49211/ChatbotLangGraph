import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

def get_weather(city: str) -> str:
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        return (
            f"🌤️ Weather in {city.title()}:\n"
            f"- Condition: {weather.capitalize()}\n"
            f"- Temp: {temp}°C (Feels like {feels_like}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind: {wind_speed} m/s"
        )
    except Exception as e:
        return f"⚠️ Failed to fetch weather: {str(e)}"

def simplify_topic(topic: str) -> str:
    try:
        prompt = f"Explain the topic '{topic}' in simple terms suitable for a beginner."
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ Failed to simplify topic: {str(e)}"
