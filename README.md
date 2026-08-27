# 🤖 AI Chatbot using Qwen2.5-0.5B-Instruct

A simple and interactive **AI Chatbot** built using a locally running Large Language Model (LLM).

This project uses **Qwen2.5-0.5B-Instruct** from Hugging Face Transformers and provides a user-friendly **Streamlit web interface** for chatting with the model.

The chatbot runs locally using **CPU inference**, making it suitable for systems without a dedicated GPU.

---

## 📌 Project Overview

The goal of this project is to build a lightweight AI chatbot that can:

- 💬 Answer user questions
- 🧠 Maintain recent conversation context
- 🤖 Generate responses using a local LLM
- 🖥️ Provide an interactive Streamlit interface
- ⚡ Run completely locally using CPU
- 🗑️ Clear the conversation history
- 🔄 Handle follow-up questions using conversation context

---

## ✨ Features

### 🤖 Local LLM

Uses:

**Qwen2.5-0.5B-Instruct**

The model is downloaded from Hugging Face and runs locally.

### 💬 Conversational Chat

The chatbot supports normal conversations and follow-up questions.

Example:

```text
User: What is Python?

AI: Python is a high-level programming language...

User: Who created it?

AI: Guido van Rossum created Python. 
```

## 🧠 Conversation Memory

The application stores recent messages and sends them as context to the model.

This allows the chatbot to understand references such as:

What is Python?
Who created it?
When was it created?

## 🖥️ Streamlit Interface

The chatbot provides a clean web-based interface using Streamlit.

## 🗑️ Clear Chat

Users can clear the current conversation using the Clear Chat button.

## ⚙️ CPU-Based Inference

The model runs using PyTorch on the CPU.

No GPU is required.

## 👩‍💻 Author

Harshvi Patel

AI/ML Engineer
