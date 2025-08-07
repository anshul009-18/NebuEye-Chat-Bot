# NebuEye 🤖

### An AI-powered application for document and general chat.

Meet NebuEye! 🚀 This application allows you to seamlessly chat with your own PDF documents and also provides a general chat assistant. It uses a combination of Google's Gemini-1.5-flash for document-based Q&A and Llama 3 (via Groq) for general conversations. It is deployed using Streamlit, providing an interactive and user-friendly interface. 📚💬

***

### 📝 Description

The NebuEye application is a **Streamlit**-based web application designed to facilitate interactive conversations. The app has two main modes:

* **Document Chat**: Users can upload multiple PDF documents. The app then processes these documents to create a searchable knowledge base, allowing users to ask questions and receive answers based *only* on the content of their uploaded files.
* **General Chat**: Users can engage in free-form conversations with an AI assistant that is not limited to the uploaded documents.

***

## Deployment

You can access the live application here:

[NebuEye Chatbot on Streamlit Cloud](https://nebueye-chat-bot.streamlit.app/))

***

### 🎯 How It Works

The application follows a structured process to deliver accurate responses:

* **PDF Loading**: The app reads multiple PDF files and extracts their text content using **PyPDF2**.
* **Text Chunking**: The extracted text is divided into smaller, manageable chunks using **RecursiveCharacterTextSplitter** from **LangChain**.
* **Vector Store Creation**: Using **GoogleGenerativeAIEmbeddings**, the application converts these text chunks into numerical vectors (embeddings). These embeddings are then stored in a local **FAISS** index, which acts as a searchable knowledge base.
* **Similarity Matching**: When a user asks a question in the "Document Chat" mode, the app compares the question's embedding with the embeddings in the FAISS index to find the most relevant text chunks.
* **Response Generation**:
    * **Document Chat**: The relevant text chunks are passed to the **Gemini-1.5-flash** model along with the user's question, and the model generates a response based on the provided context.
    * **General Chat**: The user's query is sent directly to the **Groq API**, which uses the `llama3-70b-8192` model to generate a response without any external document context.

***

### 🎯 Key Features

* **Dual Functionality**: Seamlessly switch between a document-specific chat agent and a general-purpose AI assistant.
* **Customizable Response Styles**: Choose between "Normal," "Concise," and "Detailed" response formats to tailor the AI's output.The "Detailed" mode provides a highly structured, teaching-style response with markdown headings, tables, and a conclusion.
* **Local Knowledge Base**: The processed documents are stored in a local **FAISS** vector store, so they don't need to be re-uploaded for subsequent sessions.
* **API Integration**: Uses **Google Generative AI** for RAG and **Groq** for general chat, leveraging powerful, high-performance language models.

***

### 🌟 Requirements

* `streamlit`: Used to build the interactive web application.
* `google-generativeai`: Provides access to Google's generative AI capabilities.
* `python-dotenv`: Loads environment variables from a `.env` file, securing API keys and other sensitive information.
* `langchain`: A framework for building LLM applications, handling tasks like text splitting, vector stores, and conversational chains.
* `PyPDF2`: A library for reading text content from PDF documents.
* `faiss-cpu`: A library for efficient similarity search and clustering of dense vectors, used for the vector store.
* `langchain_google_genai`: Provides the integration between LangChain and Google's generative AI models and embeddings.
* `groq`: The client for the Groq API, used to access the Llama 3 model for general chat.

***

### ▶️ Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/NebuEye.git](https://github.com/your-username/NebuEye.git)
    cd NebuEye
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up API keys**: Create a `.env` file in the root directory of the project and add your API keys for Google and Groq.
    ```env
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
    GROQ_API_KEY=YOUR_GROQ_API_KEY
    ```
4.  **Run the application**:
    ```bash
    streamlit run app.py
    ```
    The application will launch in your default web browser.

***

### 💡 Usage

* **Navigation**: Use the sidebar to switch between "General Chat" and "Document Chat".
* **Document Chat**:
    1.  Go to the "Document Chat" page.
    2.  Upload one or more PDF files.
    3.  Click the **"Process Documents"** button. The application will create a local vector store.
    4.  Once processed, you can ask questions about the documents using the chat interface.
* **General Chat**:
    1.  Go to the "General Chat" page.
    2.  Type your question or query into the text input field.
    3.  The chatbot will provide a response using the Llama 3 model.
* **Response Style**: Use the dropdown menu in the chat input area to select your preferred response style (Normal, Concise, or Detailed).
