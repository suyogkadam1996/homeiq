from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})   # only check the single top result

# A small, honest test set: real questions with a known correct answer to check for
test_questions = [
    {"question": "What is the monthly HOA fee?", "expected_answer_contains": "$340"},
    {"question": "Is short-term rental allowed?", "expected_answer_contains": "not permitted"},
    {"question": "What's the average sale price on Maple Street?", "expected_answer_contains": "$410,000"},
]

hits = 0
for case in test_questions:
    top_result = retriever.invoke(case["question"])[0]
    found = case["expected_answer_contains"] in top_result.page_content
    hits += found
    print(f"Q: {case['question']}")
    print(f"   Expected to find: '{case['expected_answer_contains']}' -> {'FOUND' if found else 'MISSING'}\n")

print(f"Retrieval accuracy: {hits}/{len(test_questions)}")