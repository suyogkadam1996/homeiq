# HomeIQ — AI Real-Estate Intelligence Platform

A hands-on project I built to learn practical AI engineering end-to-end — from a basic API call all the way to a deployed, working product.

## What it does

HomeIQ is a small AI-powered toolkit for real-estate investors, built around three features:

- **Chat Assistant** — a conversational AI that answers general real-estate questions, with streaming responses and conversation memory.
- **Property Value Estimator** — an XGBoost model trained on real product-listing data, comparing traditional ML, a from-scratch neural network, and fine-tuning approaches (see `sprint6_*.py` for the full comparison).
- **Document Assistant** — a RAG (Retrieval-Augmented Generation) pipeline that answers questions grounded in real documents (HOA agreements, inspection reports, market notes) instead of guessing.

## Why I built it this way

Each part of this project maps to a specific AI engineering skill:

| Feature | Techniques used |
|---|---|
| Chat Assistant | OpenAI API, streaming, tool calling |
| Value Estimator | Data cleaning, baseline models, traditional ML (Linear Regression, Random Forest, XGBoost), neural networks (PyTorch) |
| Document Assistant | Chunking, embeddings, vector search (LangChain + Chroma), retrieval evaluation |
| (Colab notebooks) | Fine-tuning an open-source model (Qwen) with LoRA/QLoRA, tracked with Weights & Biases |
| Deployment | Serverless deployment with Modal |

I deliberately kept the honest results in here too — including approaches that *didn't* work as well as expected (Linear Regression underperformed a naive baseline; a first neural network attempt needed input scaling to train properly). I think that's more useful to show than a project where everything "just worked."

## Running it locally

1. Clone this repo and install [uv](https://docs.astral.sh/uv/)
2. Run `uv sync` to install dependencies
3. Create a `.env` file with your own `OPENAI_API_KEY`
4. Run `uv run python sprint8_client_demo.py`

## Project structure

- `main.py` — first working API call, chained calls, streaming
- `sprint2_*.py` — Gradio UI, tool calling
- `sprint5_*.py` — RAG pipeline, retrieval evaluation
- `sprint6_*.py` — price prediction: baselines through neural networks
- `sprint8_*.py` — Modal deployment, unified client demo