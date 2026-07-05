import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Step 1: a real Python function — this is HomeIQ's actual (stand-in) sales database lookup
def get_comparable_sales(neighborhood):
    sales = {
        "maple street": "3 homes sold in the last 60 days, average price $410,000",
        "oak avenue": "2 homes sold in the last 60 days, average price $525,000",
    }
    return sales.get(neighborhood.lower(), "No recent comparable sales found for that area.")

# Step 2: describe that function to the AI, so it knows this capability exists
tools = [{
    "type": "function",
    "function": {
        "name": "get_comparable_sales",
        "description": "Get recent comparable home sales for a given neighborhood or street",
        "parameters": {
            "type": "object",
            "properties": {"neighborhood": {"type": "string"}},
            "required": ["neighborhood"]
        }
    }
}]

def main():
    messages = [{"role": "user", "content": "What have comparable homes on Maple Street sold for recently?"}]

    # Step 3: send the question, along with the list of tools the AI is allowed to request
    response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

    if response.choices[0].message.tool_calls:
        call = response.choices[0].message.tool_calls[0]
        args = json.loads(call.function.arguments)
        print(f"AI requested: {call.function.name}({args})")

        # Step 4: YOUR code actually runs the real lookup — the AI never touches this directly
        real_result = get_comparable_sales(args["neighborhood"])
        print(f"Real result from our database: {real_result}")

        # Step 5: hand the real result back, so the AI can finish its answer using true data
        messages.append(response.choices[0].message)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": real_result})
        final = client.chat.completions.create(model="gpt-4o", messages=messages)
        print(f"\nFinal answer: {final.choices[0].message.content}")
    else:
        print(response.choices[0].message.content)

if __name__ == "__main__":
    main()