from datetime import datetime
import os
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Chat setup
chat_start_time = datetime.now()
chat_history = []

print("Welcome to SearchMind!")
print("Type 'history' to see previous messages.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if user_input.lower() == "history":
        print("\nChat history:")

        for ind, msg in enumerate(chat_history):
            print(
                f"{ind+1}. "
                f"[{msg['timestamp'].strftime('%H:%M:%S')}] "
                f"{msg['role'].capitalize()}: {msg['content']}"
            )

        continue

    if not user_input:
        continue

    # Store user message
    chat_history.append({
        "timestamp": datetime.now(),
        "role": "user",
        "content": user_input
    })

    try:
        # Convert history into Gemini's message format
        messages = [
            {
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            }
            for msg in chat_history
        ]

        # Send conversation to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages
        )

        bot_response = response.text

        if not bot_response:
            print("Bot: No response generated.")
            continue

        print(f"\nBot: {bot_response}\n")

        # Store assistant response
        chat_history.append({
            "timestamp": datetime.now(),
            "role": "model",
            "content": bot_response
        })

    except Exception as e:
        print(f"Error: {e}")

print(f"\nChat started at {chat_start_time}")
print(f"Total messages: {len(chat_history)}")