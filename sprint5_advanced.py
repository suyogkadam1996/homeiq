from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

if os.path.exists("vector_db"):
    shutil.rmtree("vector_db")

loader = DirectoryLoader("homeiq_documents/", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="vector_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini")

# ---------- PART 1: Query rewriting ----------
print("=" * 60)
print("PART 1: QUERY REWRITING")
print("=" * 60)

vague_question = "What about the other one?"   # deliberately unclear, like a real follow-up

rewrite_prompt = f"""Rewrite this vague question into a clear, self-contained question about real estate.
If it's too vague to guess, make a reasonable assumption and state it.

Vague question: {vague_question}"""

rewritten = llm.invoke(rewrite_prompt).content
print(f"Original (vague): {vague_question}")
print(f"Rewritten (clear): {rewritten}\n")

print("--- Retrieval using the VAGUE question ---")
for chunk in retriever.invoke(vague_question):
    print(f"- {chunk.page_content[:60]}...")

print("\n--- Retrieval using the REWRITTEN question ---")
for chunk in retriever.invoke(rewritten):
    print(f"- {chunk.page_content[:60]}...")

# ---------- PART 2: Re-ranking ----------
print("\n" + "=" * 60)
print("PART 2: RE-RANKING")
print("=" * 60)

question = "How much does the property cost per month, not counting the mortgage?"
initial_results = retriever.invoke(question)

print(f"Question: {question}")
print("\n--- Initial order (from fast vector search) ---")
for i, chunk in enumerate(initial_results):
    print(f"{i+1}. {chunk.page_content[:60]}...")

rerank_prompt = f"""Question: {question}

Here are {len(initial_results)} retrieved passages. Rank them from MOST to LEAST relevant
to actually answering the question. Reply with just the passage numbers in order, comma-separated.

""" + "\n\n".join([f"[{i+1}] {c.page_content}" for i, c in enumerate(initial_results)])

reranked_order = llm.invoke(rerank_prompt).content
print(f"\n--- AI's re-ranked order ---")
print(reranked_order)