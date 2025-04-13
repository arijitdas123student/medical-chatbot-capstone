from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify, session
) # Added session
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import pickle
from functools import wraps # To create decorators for login check

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.static_folder = 'static'
# !!! IMPORTANT: Change this secret key in a real application! !!!
# Use a strong, random key and store it securely (e.g., environment variable)
app.secret_key = 'a-very-insecure-secret-key-change-me'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging (same as before)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hardcoded Credentials (FOR DEMO ONLY - NOT SECURE) ---
VALID_USERNAME = "user@example.com" # Or just 'user'
VALID_PASSWORD = "password123"
# -----------------------------------------------------------

# --- AI and Embeddings Configuration (same as before) ---
# Ensure GOOGLE_API_KEY is set
try:
    my_api_key_gemini = os.getenv('GOOGLE_API_KEY', None) # Provide default None
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

# --- Vector Store and Embeddings (same as before) ---
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
# --- End AI/Embeddings Config ---


# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function
# --------------------------------

# --- Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to main page
    if session.get('logged_in'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # --- IMPORTANT: Basic validation (replace with proper checks) ---
        if not username or not password:
             flash('Both username and password are required.', 'danger')
             return render_template('login.html')
        # -----------------------------------------------------------------

        # --- Check Credentials (Insecure Hardcoded Check) ---
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            session['username'] = username # Store username for display
            flash('Login successful!', 'success')
            logger.info(f"User '{username}' logged in successfully.")
            # Redirect to the page they were trying to access, or default to index
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            logger.warning(f"Failed login attempt for username: '{username}'")
            return render_template('login.html') # Show login page again on failure

    # For GET request, just show the login form
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear session variables
    logged_out_user = session.pop('username', 'User') # Get username before popping
    session.pop('logged_in', None)
    flash('You have been logged out successfully.', 'info')
    logger.info(f"User '{logged_out_user}' logged out.")
    return redirect(url_for('login'))

# --- Protect the Main Chat Page ---
@app.route('/')
@login_required # Add this decorator here
def index():
    kb_ready = vector_store is not None
    current_user = session.get('username', 'User') # Get username from session
    # Pass username to the template
    return render_template('index.html', kb_ready=kb_ready, username=current_user)

# --- Protect the Upload Functionality ---
@app.route('/upload', methods=['POST'])
@login_required # Also protect upload
def upload():
    global vector_store
    if embeddings is None:
        flash("Embeddings model is not available. Cannot process PDFs.", "danger")
        logger.error("Upload cancelled: Embeddings model not loaded.")
        return redirect(url_for('index'))

    # --- Rest of your existing upload logic ---
    # (No changes needed inside the function itself, just the decorator)
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
            if file and file.filename and file.filename.lower().endswith('.pdf'):
                try:
                    # Consider using secure_filename
                    # from werkzeug.utils import secure_filename
                    # filename = secure_filename(file.filename)
                    filename = file.filename
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    logger.info(f"Saved uploaded file: {filename}")

                    pdf_loader = PyPDFLoader(file_path)
                    loaded_docs = pdf_loader.load()
                    documents.extend(loaded_docs)
                    processed_files.append(filename)
                    logger.info(f"Loaded {len(loaded_docs)} pages/documents from {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    errors.append(f"Could not process {file.filename}: {str(e)[:100]}...")
            elif file and file.filename:
                 logger.warning(f"Skipped non-PDF file: {file.filename}")
                 errors.append(f"Skipped non-PDF file: {file.filename}")


        if not documents:
            error_message = "No valid PDF documents were processed."
            if errors: error_message += " Issues: " + "; ".join(errors)
            flash(error_message, "warning")
            return redirect(url_for('index'))

        logger.info(f"Attempting to create/update vector store with {len(documents)} documents.")
        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(documents, embeddings)
                logger.info("Created new FAISS vector store.")
            else:
                vector_store.add_documents(documents)
                logger.info("Added documents to existing FAISS vector store.")

            with open(vector_store_path, 'wb') as f:
                pickle.dump(vector_store, f)
            logger.info(f"Saved updated vector store to {vector_store_path}")

        except Exception as e:
            logger.error(f"Error creating/updating/saving vector store: {e}")
            flash("PDFs processed, but failed to update or save the knowledge base.", "danger")

        flash_message = f"Processed {len(processed_files)} PDF(s): {', '.join(processed_files)}. Knowledge base updated."
        category = "success"
        if errors:
            flash_message += " Some files had issues: " + "; ".join(errors)
            category = "warning" # Mark as warning if there were errors
        flash(flash_message, category)
        return redirect(url_for('index'))

    except Exception as e:
        logger.exception("An unexpected error occurred during PDF upload/processing.")
        flash(f"An unexpected error occurred during upload: {str(e)[:100]}...", "danger")
        return redirect(url_for('index'))


# --- Protect the Ask Functionality ---
@app.route('/ask', methods=['POST'])
@login_required # Protect the chat endpoint
def ask():
    if vector_store is None:
        logger.warning("Chat attempt made before knowledge base (vector store) is ready.")
        return jsonify({"error": "Knowledge base is not ready. Please upload PDF documents first."}), 400

    if model is None:
         logger.error("Chat attempt failed: Generative AI model not configured.")
         return jsonify({"error": "Chat functionality is currently unavailable due to configuration issues."}), 503

    question = request.form.get('prompt', '').strip()
    if not question:
         logger.warning("Empty question received.")
         return jsonify({"error": "Question cannot be empty."}), 400

    # --- Rest of your existing ask logic (no changes needed here) ---
    logger.info(f"Received question from user '{session.get('username', 'Unknown')}': '{question}'")
    try:
        logger.info("Performing similarity search...")
        relevant_docs = vector_store.similarity_search(question, k=3)
        context = "\n---\n".join([doc.page_content for doc in relevant_docs])
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents for context.")
        max_context_length = 5000
        if len(context) > max_context_length:
             context = context[:max_context_length] + "\n... (context truncated)"
             logger.warning("Context truncated due to length limit.")

        custom_prompt = f"""You are a helpful medical assistant chatbot named MedBot... (rest of prompt same as before)

        Context from documents:
        ---
        {context}
        ---

        User Question: {question}

        Answer:"""

        logger.info("Generating response from AI model...")
        response = model.generate_content(custom_prompt)

        answer = "Sorry, I could not generate a response at this time."
        status_code = 500

        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            # ... (handling blocked response)
            block_reason = response.prompt_feedback.block_reason
            logger.warning(f"AI response blocked for user '{session.get('username', 'Unknown')}'. Reason: {block_reason}")
            answer = f"The response was blocked due to safety concerns ({block_reason}). Please rephrase."
            status_code = 400
        elif response and hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text') and response.parts[0].text:
            # ... (handling successful response)
            answer = response.parts[0].text
            logger.info(f"AI Response generated for '{session.get('username', 'Unknown')}': '{answer[:100]}...'")
            status_code = 200
        elif response and hasattr(response, 'text') and response.text:
             # ... (fallback handling)
             answer = response.text
             logger.info(f"AI Response generated (using .text) for '{session.get('username', 'Unknown')}': '{answer[:100]}...'")
             status_code = 200
        else:
            logger.error(f"Unexpected response format for user '{session.get('username', 'Unknown')}': {response}")

        return jsonify(answer if status_code == 200 else {"error": answer}), status_code

    except Exception as e:
        logger.exception(f"Error during chat interaction for user '{session.get('username', 'Unknown')}'.")
        return jsonify({"error": "An internal server error occurred while processing your request."}), 500


# --- Error Handlers (Keep 404, maybe add others) ---
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 Not Found: {request.url}")
    # Redirect to login if not logged in, otherwise show a simple 404 message or template
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return "<h2>404 - Page Not Found</h2><p>The requested URL was not found.</p><a href='/'>Go to Chat</a>", 404

# Add more specific error handlers if needed (e.g., 500)
@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"500 Internal Server Error: {e}")
    # Avoid showing detailed errors to users in production
    return "<h2>500 - Internal Server Error</h2><p>Sorry, something went wrong on our end.</p>", 500


# --- Run the App ---
if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000) # Keep debug=True for development only
# Add this new route function in your app.py

@app.route('/new_chat')
@login_required # Ensure user is logged in
def new_chat():
    """Renders the clean 'New Chat' starting page."""
    current_user = session.get('username', 'User') # Get username from session
    # Pass username and an identifier for the active page
    return render_template('new_chat.html', username=current_user, active_page='new_chat')

# --- Modify the existing index route ---
@app.route('/')
@login_required
def index():
    kb_ready = vector_store is not None
    current_user = session.get('username', 'User')
    # Pass username, kb_ready, and active page identifier
    return render_template('index.html', kb_ready=kb_ready, username=current_user, active_page='home') # Add active_page

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    logger.info(f"--- Entered /ask route with POST method ---") # Log entry
    logger.info(f"Session username: {session.get('username', 'Unknown')}")
    logger.info(f"Vector store exists: {vector_store is not None}")
    logger.info(f"Model exists: {model is not None}")

    if vector_store is None:
        logger.warning("Chat attempt made before knowledge base (vector store) is ready.")
        return jsonify({"error": "Knowledge base is not ready. Please upload PDF documents first."}), 400

    if model is None:
        logger.error("Chat attempt failed: Generative AI model not configured.")
        return jsonify({"error": "Chat functionality is currently unavailable due to configuration issues."}), 503

    question = request.form.get('prompt', '').strip()
    logger.info(f"Received question: '{question}'") # Log the question

    if not question:
        logger.warning("Empty question received.")
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        logger.info("Performing similarity search...")
        relevant_docs = vector_store.similarity_search(question, k=3)
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")
        if not relevant_docs:
            logger.warning("No relevant documents found by similarity search.")
            # Decide how to handle this - maybe answer without context or inform the user
            context = "No relevant information found in the uploaded documents."
        else:
             context = "\n---\n".join([doc.page_content for doc in relevant_docs])
             logger.info(f"Context snippet: {context[:200]}...") # Log beginning of context

        # ... (rest of context processing and prompt building) ...
        logger.info("Context length: " + str(len(context))) # Check context length

        custom_prompt = f"""You are a helpful medical assistant chatbot named MedBot... (rest of prompt same as before)

        Context from documents:
        ---
        {context}
        ---

        User Question: {question}

        Answer:"""
        logger.info("Generating response from AI model...")
        response = model.generate_content(custom_prompt)
        logger.info("Raw AI response received.") # Log that response was received

        # ... (rest of response handling) ...
        logger.info(f"Final answer prepared: '{answer[:100]}...'")
        return jsonify(answer if status_code == 200 else {"error": answer}), status_code

    except Exception as e:
        logger.exception(f"Error during chat interaction for user '{session.get('username', 'Unknown')}'.") # Use logger.exception
        # Return a more specific error message if possible
        return jsonify({"error": f"An internal server error occurred in /ask: {str(e)}"}), 500

