# ./my_RAG/rag.py

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

# docs
docs = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Tokyo is the capital of Japan."
]
model = SentenceTransformer("BAAI/bge-base-en")
# bge-base-en is a 768 dimensional English sentence embedding model,
# trained for semantic similarity and retrieval tasks.
embeddings = model.encode(docs, normalize_embeddings=True)
# Each embedding vector is L2-normalized, which allows dot product = cosine similarity

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q_emb), top_k)
    return [docs[i] for i in ids[0]], scores[0]

gen_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# RAG
def rag_generate(query, top_k=3):
    retrieved_docs, scores = retrieve(query, top_k)
    context = "\n".join(f"Doc{i+1}: {d}" for i, d in enumerate(retrieved_docs))
    prompt = f"Question: {query}\nRelevant Information:\n{context}\nAnswer:"

    inputs = gen_tok(prompt, return_tensors='pt')
    outputs = gen_model.generate(**inputs, max_new_tokens=128)
    return gen_tok.decode(outputs[0], skip_special_tokens=True), retrieved_docs


ans, retrieved = rag_generate("Where is the Eiffel Tower located?")
print("Answer:", ans)
print("Retrieved context:", retrieved)