from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

raw_listing_text = """
3BR/2BA fixer needing TLC, huge backyard, close 2 schools, motivated seller,
asking 415k, roof redone 2019, some water damage in basement reported by prior owner
"""

def main():
    # Call 1: pull out the key facts as a short, clean note
    facts_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract price, size, and location as a short bullet list:\n\n{raw_listing_text}"}]
    )
    clean_facts = facts_response.choices[0].message.content
    print("--- CLEAN FACTS ---")
    print(clean_facts)

    # Call 2: use that clean note to stream a polished pitch paragraph
    print("\n--- STREAMING PITCH ---")
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Write a short, appealing pitch paragraph from these facts:\n\n{clean_facts}"}],
        stream=True
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        print(piece, end="", flush=True)
    print()

if __name__ == "__main__":
    main()