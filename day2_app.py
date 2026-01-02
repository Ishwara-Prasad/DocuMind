import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize model and DB
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

# Get or create collection safely
try:
    collection = client.get_collection("linux_docs")
except:
    collection = client.create_collection("linux_docs")

# Add sample documents (only if not already added)
if collection.count() == 0:
    docs = [
        "To restart Apache in cPanel, use WHM service manager.",
        "You can create email accounts in cPanel under the Email section.",
        "WHM allows you to manage multiple hosting accounts."
    ]
    embeddings = model.encode(docs).tolist()
    collection.add(documents=docs, embeddings=embeddings, ids=["1","2","3"])

# Streamlit UI
st.title("DocuMind - Day 2")
query = st.text_input("Ask me something about cPanel/WHM:")

if query:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=2)
    st.write("### Query:", query)
    st.write("### Results:")
    for doc in results["documents"][0]:
        st.write("-", doc)
