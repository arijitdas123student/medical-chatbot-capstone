from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, session
)
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import pickle
from functools import wraps

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.static_folder = 'static'
app.secret_key = 'a-very-insecure-secret-key-change-me'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_USERNAME = "user@example.com"
VALID_PASSWORD = "password123"

try:
    my_api_key_gemini = os.getenv('GOOGLE_API_KEY', None)
    if not my_api_key_gemini:
        logger.warning("GOOGLE_API_KEY environment variable not set. AI features may fail.")
        model = None
    else:
        genai.configure(api_key=my_api_key_gemini)
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Google Generative AI configured successfully.")
except Exception as e:
    logger.error(f"Error configuring Google Generative AI: {e}")
    model = None

vector_store = None
vector_store_path = 'vector_store.pkl'
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = None
try:
    if os.path.exists(vector_store_path):
        with open(vector_store_path, 'rb') as f:
            vector_store = pickle.load(f)
        logger.info(f"Loaded existing vector store from {vector_store_path}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    logger.info(f"HuggingFace Embeddings model '{embedding_model_name}' loaded.")
except Exception as e:
    logger.error(f"Error loading vector store or embeddings: {e}")
    vector_store = None
    embeddings = None


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Both username and password are required.', 'danger')
            return render_template('login.html')
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            logger.info(f"User '{username}' logged in successfully.")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            logger.warning(f"Failed login attempt for username: '{username}'")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logged_out_user = session.pop('username', 'User')
    session.pop('logged_in', None)
    flash('You have been logged out successfully.', 'info')
    logger.info(f"User '{logged_out_user}' logged out.")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    kb_ready = vector_store is not None
    current_user = session.get('username', 'User')
    return render_template('index.html', kb_ready=kb_ready, username=current_user)

@app.route('/new_chat')
@login_required
def new_chat():
    current_user = session.get('username', 'User')
    return render_template('newchat.html', username=current_user, active_page='new_chat')

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    global vector_store
    if embeddings is None:
        flash("Embeddings model is not available. Cannot process PDFs.", "danger")
        logger.error("Upload cancelled: Embeddings model not loaded.")
        return redirect(url_for('index'))

    try:
        if 'pdf_files' not in request.files:
            flash("No file part in the request.", "warning")
            return redirect(url_for('index'))

        files = request.files.getlist('pdf_files')
        if not files or (len(files) == 1 and files[0].filename == ''):
            flash("No files selected for upload.", "warning")
            return redirect(url_for('index'))

        documents = []
        processed_files = []
        errors = []

        for file in files:
            if file and file.filename.lower().endswith('.pdf'):
                try:
                    filename = file.filename
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    pdf_loader = PyPDFLoader(file_path)
                    documents.extend(pdf_loader.load())
                    processed_files.append(filename)
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    errors.append(f"Could not process {file.filename}: {str(e)[:100]}.")

        if not documents:
            error_message = "No valid PDF documents were processed."
            if errors:
                error_message += " Issues: " + "; ".join(errors)
            flash(error_message, "warning")
            return redirect(url_for('index'))

        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(documents, embeddings)
            else:
                vector_store.add_documents(documents)

            with open(vector_store_path, 'wb') as f:
                pickle.dump(vector_store, f)
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            flash("Processed PDFs, but failed to save knowledge base.", "danger")

        flash(f"Processed {len(processed_files)} PDFs successfully.", "success")
        return redirect(url_for('index'))

    except Exception as e:
        logger.exception("Unexpected error during upload")
        flash(f"Unexpected error: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    try:
        if model is None:
            return jsonify({"error": "Gemini model not loaded."}), 503
        if vector_store is None:
            return jsonify({"error": "Knowledge base not ready. Upload PDFs first."}), 400

        question = request.form.get('prompt', '').strip()
        if not question:
            return jsonify({"error": "Question cannot be empty."}), 400

        relevant_docs = vector_store.similarity_search(question, k=3)
        context = "\n---\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""You are a helpful medical assistant chatbot named MedBot.

        Context:
        {context}

        User Question: {question}

        Answer:"""

        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            answer = response.text
        elif hasattr(response, 'parts') and response.parts:
            answer = response.parts[0].text
        else:
            answer = "Gemini did not return a valid response."

        return jsonify(answer), 200

    except Exception as e:
        logger.exception("Exception in /ask")
        return jsonify({"error": "Internal server error."}), 500

@app.errorhandler(404)
def not_found(e):
    return "<h2>404 - Page Not Found</h2><a href='/'>Go Home</a>", 404

@app.errorhandler(500)
def internal_error(e):
    return "<h2>500 - Internal Server Error</h2><p>Please try again later.</p>", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
