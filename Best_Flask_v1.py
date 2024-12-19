from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import os
from flask_cors import CORS



app = Flask(__name__)

# Enable CORS
CORS(app)

CORS(app, resources={r"/*": {"origins": "*"}})


# Model paths and configuration
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Define the local path to the model
# local_model_path = "C:/Users/sachi/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"

# # Load the HuggingFace model embeddings from the local path
# embeddings = HuggingFaceEmbeddings(model_name=local_model_path)

# Load model and vectorstore once during startup
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        device="cpu",  # Specify GPU device
        max_new_tokens=128,
        temperature=0.5
    )



# def load_llm():
    

# Custom Prompt Template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval for each vectorstore"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 1}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# QA Model Function
def qa_bot(query):
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    
    response = qa({'query': query})
    return response['result']

# Flask route to handle QA queries
@app.route('/ask', methods=['POST'])
def ask_bot():
    data = request.json
    query = data.get("query")
    print(query,"798465132**********************")

    # Ensure we have a valid query
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Run the query through the QA bot
        answer = qa_bot(query)
        # Return the answer
        return jsonify({
            "answer": answer,
        })

    except Exception as e:
        # Handle exceptions
        return jsonify({"error": str(e)}), 500

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Flask App Runner
if __name__ == '__main__':
    # Check if FAISS vectorstore exists
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"FAISS vectorstore not found at {DB_FAISS_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=True)
