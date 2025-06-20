import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from helpers.knowledge import get_weather, simplify_topic

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# ----------------------
# 🧠 Graph State
class GraphState(TypedDict):
    input: str
    output: str

# ----------------------
# 🧭 Router
def router(state: GraphState) -> dict:
    text = state["input"].lower()
    if "weather" in text or "temperature" in text:
        return {"next": "weather"}
    return {"next": "simplify"}

# ----------------------
def extract_city(user_input: str) -> str:
    try:
        prompt = (
            f"You are a smart assistant. Extract just the city name from the following query: '{user_input}'. "
            "Even if the sentence is incomplete, short form, or slightly misspelled, return the most likely city name."
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return user_input.strip() # Fallback to original input if extraction fails
# 🌤️ Weather Node
def weather_node(state: GraphState) -> GraphState:
    user_input = state["input"]
    city = extract_city(user_input)
    weather_info = get_weather(city)
    return {"input": city, "output": weather_info}

# ----------------------
# 📘 Simplifier Node
def simplify_node(state: GraphState) -> GraphState:
    topic = state["input"]
    result = simplify_topic(topic)
    return {"input": topic, "output": result}

# ----------------------
# 🚧 Build Graph
builder = StateGraph(GraphState)

builder.add_node("router", router)
builder.add_node("weather", weather_node)
builder.add_node("simplify", simplify_node)

builder.set_entry_point("router")

# 🧩 Transition based on router's return value
builder.add_conditional_edges(
    "router",
    lambda x: x["next"],  # This selects "weather" or "simplify"
    {
        "weather": "weather",
        "simplify": "simplify"
    }
)

builder.add_edge("weather", END)
builder.add_edge("simplify", END)

graph = builder.compile()
