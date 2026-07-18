import modal

app = modal.App("homeiq-live-demo")

image = modal.Image.debian_slim().pip_install(
    "gradio", "openai", "fastapi", "datasets", "scikit-learn", "xgboost",
    "langchain", "langchain-openai", "langchain-chroma", "langchain-text-splitters", "chromadb"
)

@app.function(image=image, secrets=[modal.Secret.from_name("OPENAI_API_KEY")], min_containers=1, timeout=600)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def serve():
    import gradio as gr
    from openai import OpenAI
    from fastapi import FastAPI

    client = OpenAI()

    # ---------- TAB 1: Chat Assistant ----------
    def answer_question(message, history):
        messages = [{"role": "system", "content": "You are HomeIQ's real-estate assistant."}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})
        stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages, stream=True)
        partial = ""
        for chunk in stream:
            partial += chunk.choices[0].delta.content or ""
            yield partial

    # ---------- TAB 2: Property Value Estimator - trained ONCE when the app starts ----------
    print("Training price model...")
    from datasets import load_dataset
    from sklearn.feature_extraction.text import CountVectorizer
    from xgboost import XGBRegressor

    dataset = load_dataset("mcauley-lab/amazon-reviews-2023", "raw_meta_All_Beauty", split="full")
    prices, descriptions = [], []
    seen = set()
    for item in dataset:
        price_str = item["price"]
        if price_str is not None and price_str != "None":
            try:
                price = float(price_str)
                if price > 0 and item["title"] not in seen:
                    seen.add(item["title"])
                    prices.append(price)
                    descriptions.append(item["title"])
            except (ValueError, TypeError):
                continue

    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(descriptions)
    price_model = XGBRegressor(random_state=42)
    price_model.fit(X, prices)
    print("Price model ready.")

    def estimate_price(description):
        if not description.strip():
            return "Please enter a description first."
        X_new = vectorizer.transform([description])
        prediction = price_model.predict(X_new)[0]
        return f"Estimated value: ${prediction:,.2f}"

    # ---------- TAB 3: Document Assistant - documents embedded directly, built ONCE at startup ----------
    print("Building document assistant...")
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_chroma import Chroma

    raw_documents = [
        Document(page_content="""Maple Street Condos HOA Agreement
Monthly HOA dues are $340 per month, covering water, trash, and building maintenance.
Short-term rentals under 6 months are not permitted under current HOA bylaws.
Pets are allowed with a one-time $200 pet deposit."""),
        Document(page_content="""Property Inspection Report - 42 Oak Street
Roof was replaced in 2019 and is in good condition.
Minor water damage was noted in the basement, reported by the prior owner. Recommend further evaluation.
Electrical panel is up to code as of the 2023 inspection."""),
        Document(page_content="""Maple Street Neighborhood Market Notes
3 comparable homes sold on Maple Street in the last 60 days, average price $410,000.
2 comparable homes sold on Oak Avenue in the last 60 days, average price $525,000.
Market trend: steady demand, slight price increase over the last quarter."""),
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_llm = ChatOpenAI(model="gpt-4o-mini")
    print("Document assistant ready.")

    def answer_from_documents(question):
        if not question.strip():
            return "Please ask a question first."
        retrieved_chunks = retriever.invoke(question)
        context = "\n\n".join([c.page_content for c in retrieved_chunks])
        prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        answer = rag_llm.invoke(prompt)
        return answer.content

    # ---------- Combine everything ----------
    with gr.Blocks(title="HomeIQ Demo") as demo:
        gr.Markdown("# HomeIQ - AI Real-Estate Intelligence Platform")
        with gr.Tab("Chat Assistant"):
            gr.ChatInterface(answer_question)
        with gr.Tab("Property Value Estimator"):
            desc_input = gr.Textbox(label="Property description")
            price_output = gr.Textbox(label="Estimated value")
            estimate_button = gr.Button("Estimate Value")
            estimate_button.click(estimate_price, inputs=desc_input, outputs=price_output)
        with gr.Tab("Document Assistant"):
            doc_question = gr.Textbox(label="Ask about HOA rules, inspections, or market data")
            doc_answer = gr.Textbox(label="Answer")
            ask_button = gr.Button("Ask")
            ask_button.click(answer_from_documents, inputs=doc_question, outputs=doc_answer)

    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo, path="/")