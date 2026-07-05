import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def answer_question(message, history):
    messages = [{"role": "system", "content": "You are HomeIQ's real-estate assistant."}]
    messages.extend(history)   # history is already a list of {"role": ..., "content": ...} dicts
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True)
    partial = ""
    for chunk in stream:
        partial += chunk.choices[0].delta.content or ""
        yield partial

gr.ChatInterface(answer_question).launch(inbrowser=True)