# ./my_RAG/my_RAG_project/rag_real.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import AutoModelForCausalLM

# load document
data_dir = "./data"
docs = []

for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)
    ext  = file.lower()
    if ext.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif ext.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    elif ext.endswith(".md"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        loader = UnstructuredFileLoader(path)
    docs.extend(loader.load())
    # loader.load() returns list of document objects, each object looks like 
    # Document(
    #       page_content="...text from this file...",
    #       metadata={"source": "path/to/file.pdf", "page": 1, ...}
    #   )

print(f"Loaded {len(docs)} documents from {data_dir}")

# split into smaller chunks for retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# breaks long documents into smaller overlapping chunks, our model has a relatively small
# context length, so we break it, this also ensures only relevant paragraphs are retrieved
# Each chunk: 1000 chars = around 200-300 tokens
chunks = splitter.split_documents(docs)
print(f"Generated {len(chunks)} chunks")


# Embed and build FAISS index
embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5") # sentence embedding model
db = FAISS.from_documents(chunks, embedder) # each chunk gets one semantic representation vector
retriever = db.as_retriever(search_kwargs={"k": 2}) # wraps it so LangChain can use


# Load generator model
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

gen_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.5,
)
llm = HuggingFacePipeline(pipeline=gen_pipe)


# Combine into Retrieval-Augmented Generator
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
# stuff: Simply concatenates all retrieved chunks into one long prompt, then sends that to the model.
# "map_reduce", "refine", "map_rerank"

print("\nType your question (or 'exit' to quit):\n")
while True:
    q = input(">> ")
    if q.strip().lower() in {"exit", "quit"}:
        break
    ans = qa.invoke({"query": q})
    print("\nAnswer:\n", ans, "\n")

