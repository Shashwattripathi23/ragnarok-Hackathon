# 🚀 Technical Docs.

Welcome to the Friday Mini-Hack! This repository contains the **Starter Baseline** to help you hit the ground running. You have 3.5 hours to turn raw university course files into a smart, personalized AI study assistant. 

## 📁 Repository Structure

* `app.py` - The Streamlit frontend. Handles the chat interface and document loading UI.
* `agent.py` - The brain. Connects to the **Groq API** and handles the AI prompts.
* `document_parser.py` - The extractor. 
* `requirements.txt` - All the necessary Python packages.
* `.env.example` - Template for your API keys (Security first!).


## 🛠️ Quick Start Guide

Get your baseline running in under 2 minutes:

### 1. Set up a Virtual Environment (Highly Recommended)
Keep your project dependencies clean!
```bash
# Create the environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Add Your Groq API Key

1. Create a new file named exactly `.env` in the root folder.
2. Add your Groq API key to it:

```text
GROQ_API_KEY=gsk_your_api_key_here

```

### 4. Launch the App!

```bash
streamlit run app.py

```

Your browser will automatically open to `http://localhost:8501`. Click **"Load Course Materials"** in the sidebar to parse the data, and start chatting!

## 🧩 Where to Hack? (Extension Points)

Here are some ideas to get started with:

* **`document_parser.py`:** Add functions to read the `.ipynb` (Jupyter Notebook) files or `.mat` data files in the Course Materials. Build a smarter "chunking" algorithm so you don't overwhelm the LLM's context window.
* **`app.py`:** Instead of loading the *entire* course into the chat context, build a search function to only pass the most relevant paragraphs to the AI.
* **`agent.py`:** Tweak the `system_prompt`. Make the agent act as a strict quiz-master, a flashcard generator, or an exam scheduler.

## ⚠️ Hackathon Rules

1. **Empower, Don't Cheat:** Build tools that make students better, not lazier. Tools designed solely to solve graded assignments will be disqualified.
2. **Safety First:** DO NOT use or hardcode University Leiden credentials anywhere. Keep your Groq API keys in the `.env` file and out of your scripts. Do not push the .env file to your repository!

Good luck, and may the best Study Agent win! 🍕🏆

---
