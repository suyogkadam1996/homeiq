from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()          # reads your .env file and makes OPENAI_API_KEY available
client = OpenAI()      # creates a connection to OpenAI, using that key automatically

def main():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a knowledgeable real-estate assistant. Be concise and factual."},
            {"role": "user", "content": "In one sentence, what should a first-time buyer check before making an offer?"}
        ]
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()