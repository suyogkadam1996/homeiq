from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

# Clear any old, possibly duplicated database before rebuilding
if os.path.exists("vector_db"):
    shutil.rmtree("vector_db")

# Step 1: load every document in the folder
loader = DirectoryLoader("homeiq_documents/", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# Step 2: break each document into smaller, focused pieces
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Step 3: convert each chunk into meaning-representing numbers, and store them
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="vector_db")
print(f"Stored {vectorstore._collection.count()} chunks, ready to search by meaning")

# Step 4: ask a real question, and retrieve the most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
question = "Can I rent out this unit short-term under the HOA rules?"
retrieved_chunks = retriever.invoke(question)

print(f"\n--- Top {len(retrieved_chunks)} relevant chunks for: '{question}' ---")
for i, chunk in enumerate(retrieved_chunks):
    print(f"\n[{i+1}] {chunk.page_content}")

# Step 5: use the retrieved chunks to generate a real, grounded answer
llm = ChatOpenAI(model="gpt-4o-mini")

context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
prompt = f"Based on this context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"

answer = llm.invoke(prompt)
print(f"\n--- FINAL ANSWER ---")
print(answer.content)