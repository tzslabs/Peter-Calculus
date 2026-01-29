import streamlit as st
import ollama
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import os
import time
import threading

# Page configuration
st.set_page_config(
    page_title="Peter Calculus",
    page_icon="∞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .bot-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    
    /* Custom loading animations */
    .stSpinner > div {
        border-top-color: #1f77b4 !important;
        border-right-color: #1f77b4 !important;
    }
    
    /* Custom loading message styling */
    .stSpinner + div {
        color: #1f77b4 !important;
        font-weight: bold !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    try:
        # Initialize local embeddings (free, runs on your computer)
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load documents
        pdf_dir = Path("sources")
        all_docs = []
        
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob("*.pdf"):
                loader = PyMuPDFLoader(str(pdf_file))
                pages = loader.load()
                full_text = "\n".join(page.page_content for page in pages)
                merged_doc = Document(page_content=full_text, metadata={"source": str(pdf_file)})
                all_docs.append(merged_doc)
        
        # Create vector store
        vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding_model)
        
        # Check if Ollama is available and has a model
        # You'll need to run: brew install ollama && ollama pull llama3.2
        model_name = "llama3.2"  # You can also use: mistral, llama2, phi, etc.
        
        return vectorstore, model_name, len(all_docs)
        
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, 0

def get_relevant_context(user_query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Only get top 2 results for speed
    relevant_docs = retriever.invoke(user_query)
    # Limit context to 2000 characters for faster processing
    context = "\n\n".join([doc.page_content[:1000] for doc in relevant_docs])
    return context[:2000]

def get_instruction(question, vectorstore):
    text_context = get_relevant_context(question, vectorstore)
    prompt = f"""
    1. You are a capable math tutor who explains calculus concepts and solves problems step by step.

    Instructions:
    1. You are in conversation with a bright but confused student.
    2. Provide clear, step by step explanations to their calculus questions.
    3. Use simple language and avoid jargon.
    4. Use an analogy or real-world example to explain complex concepts.
    5. If you don't know the answer, say "I don't know" instead of making up an answer.
    6. **IMPORTANT**: Format all mathematical expressions using LaTeX syntax. Use $...$ for inline math and $$...$$ for display math. For example:
       - Write integrals as: $\\int f(x) dx$
       - Write fractions as: $\\frac{{1}}{{x}}$
       - Write derivatives as: $\\frac{{dy}}{{dx}}$
       - Write limits as: $\\lim_{{x \\to a}} f(x)$
    
    
    User query: {question}
    """
    
    text_contexts= get_relevant_context(question, vectorstore)
    prompt += f"""Most relevant information or context: ```{text_contexts}```

    """
    return prompt

def generate_response(prompt, model_name):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.7,
                'num_predict': 500,  # Limit response length for speed
                'top_k': 20,
                'top_p': 0.9
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running and you have the model installed."

def dynamic_loading_screen(loading_messages, initialize_func):
    import random
    
    # Create placeholders for the loading interface
    loading_container = st.empty()
    
    # Shuffle the messages for variety
    messages = loading_messages.copy()
    random.shuffle(messages)
    
    # Initialize the result container
    result = [None]  # Use list to make it mutable for the thread
    exception_container = [None]
    
    def run_initialization():
        try:
            result[0] = initialize_func()
        except Exception as e:
            exception_container[0] = e
    
    # Start the initialization in a separate thread
    init_thread = threading.Thread(target=run_initialization)
    init_thread.start()
    
    # Cycle through loading messages while initialization runs
    message_index = 0
    while init_thread.is_alive():
        current_message = messages[message_index % len(messages)]
        
        with loading_container.container():
            # Create a clean loading display without the blue box
            st.markdown(f"""
            <div style="text-align: center; padding: 30px;">
                <div style="margin-bottom: 20px;">
                    <div style="border: 4px solid #f3f3f3; border-top: 4px solid #1f77b4; 
                                border-radius: 50%; width: 50px; height: 50px; 
                                animation: spin 1s linear infinite; margin: 0 auto;"></div>
                </div>
                <div style="font-size: 1.4rem; color: #1f77b4; font-weight: bold; margin-bottom: 20px;">
                    {current_message}
                </div>
            </div>
            <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            """, unsafe_allow_html=True)
        
        time.sleep(2.5)  # Change message every 1.5 seconds
        message_index += 1
    
    # Wait for the thread to complete
    init_thread.join()
    
    # Clear the loading display
    loading_container.empty()
    
    # Check if there was an exception
    if exception_container[0]:
        raise exception_container[0]
    
    return result[0]

def main():
    # Header with mathematical symbol
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align: center; font-size: 4rem; color: #1f77b4; margin-bottom: 1rem;">∞</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-header">Peter Calculus</div>', unsafe_allow_html=True)
    
    # Define loading messages
    loading_messages = [
        "🔧 Setting up Peter's math workshop...",
        "📚 Loading calculus knowledge base...",
        "⚙️ Preparing your personal math tutor...",
        "🧠 Calibrating mathematical genius...",
        "🍺 Cracking open a cold one with the equations...",
        "🎯 Loading Peter's best dad jokes about math...",
        "🎪 Training mathematical circus performers...",
        "🔮 Consulting the calculus crystal ball...",
        "🎸 Tuning up for some math rock...",
        "🍕 Preparing brain food (pizza-shaped integrals)...",
        "🎲 Rolling for critical thinking...",
        "🎭 Rehearsing mathematical performances...",
        "🚀 Launching into mathematical orbit...",
        "🎬 Directing the calculus movie..."
    ]
    
    # Initialize components with dynamic loading
    try:
        vectorstore, model_name, doc_count = dynamic_loading_screen(loading_messages, initialize_components)
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        return
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("<h3 style='text-align: center;'>💡 Sample Questions</h3>", unsafe_allow_html=True)
        
        # Full list of 40 sample questions
        all_sample_questions = [
            "What is the integral of 1/x dx?",
            "How do I solve derivatives using chain rule?",
            "Explain the fundamental theorem of calculus",
            "What are the basic integration techniques?",
            "How do I find the derivative of sin(x)?",
            "What is integration by parts?",
            "Explain the concept of limits",
            "How do I solve u-substitution problems?",
            "What is the derivative of e^x?",
            "How do I find critical points?",
            "What is the second derivative test?",
            "How do I solve related rates problems?",
            "What is the mean value theorem?",
            "How do I find area under curves?",
            "What is partial fraction decomposition?",
            "How do I solve optimization problems?",
            "What is L'Hôpital's rule?",
            "How do I find inflection points?",
            "What is the squeeze theorem?",
            "How do I solve differential equations?",
            "What is the derivative of ln(x)?",
            "How do I find the limit of a function?",
            "What is trigonometric substitution?",
            "How do I solve implicit differentiation?",
            "What is the product rule for derivatives?",
            "How do I find local maximum and minimum?",
            "What is the quotient rule?",
            "How do I evaluate improper integrals?",
            "What is the intermediate value theorem?",
            "How do I find the derivative of inverse functions?",
            "What is integration by trigonometric substitution?",
            "How do I solve separable differential equations?",
            "What is the power rule for derivatives?",
            "How do I find asymptotes of a function?",
            "What is the ratio test for series?",
            "How do I solve parametric equations?",
            "What is the root test for convergence?",
            "How do I find the centroid of a region?",
            "What is Taylor series expansion?",
            "How do I solve polar coordinate problems?",
            "What is the comparison test for series?",
            "How do I find arc length of curves?",
            "What is the alternating series test?",
            "How do I solve volume of revolution problems?",
            "What is the integral test for series?",
            "How do I find surface area of revolution?",
            "What is partial differentiation?",
            "How do I solve Lagrange multipliers?",
            "What is the divergence theorem?",
            "How do I evaluate line integrals?"
        ]
        
        # Randomly select 6 questions to display
        import random
        if "displayed_questions" not in st.session_state:
            st.session_state.displayed_questions = random.sample(all_sample_questions, 6)
        
        for question in st.session_state.displayed_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.current_question = question
        
        st.markdown("---")
        st.markdown(f"""
        <div style="background-color: #e3f2fd; border: 2px solid #1976d2; border-radius: 10px; padding: 15px; margin: 10px 0;">
            <h3 style='text-align: center; margin-top: 0; margin-bottom: 15px; color: #1976d2;'>⚙️ Model Information</h3>
            <p><strong>📄 Documents loaded:</strong> {doc_count}</p>
            <p><strong>🧠 Model:</strong> {model_name} (Ollama - Local & Free)</p>
            <p><strong>🔍 Embeddings:</strong> all-MiniLM-L6-v2 (Local)</p>
            <p><strong>💰 Cost:</strong> $0.00 - Runs entirely on your computer!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            # Render the response with LaTeX support
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me any calculus question...")
    
    # Handle sample question selection
    if "current_question" in st.session_state:
        user_input = st.session_state.current_question
        del st.session_state.current_question
    
    if user_input:
        # Display user message
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_input}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                # Build conversation context with chat history
                conversation = "\n\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in st.session_state.messages[-6:]  # Keep last 6 messages for context
                ])
                
                # Get prompt with current question
                current_prompt = get_instruction(user_input, vectorstore)
                
                # Combine history with current question if there's history
                if conversation:
                    full_prompt = f"Previous conversation:\n{conversation}\n\nCurrent question:\n{current_prompt}"
                else:
                    full_prompt = current_prompt
                
                answer = generate_response(full_prompt, model_name)
                print("answer:", answer)

                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                print("st.session_state.messages:", st.session_state.messages)
                
                # Display bot response with LaTeX support
                st.markdown(answer, unsafe_allow_html=True)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
        
        # Rerun to update the display
        st.rerun()
    
    # Clear chat button - only show if there are messages
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()