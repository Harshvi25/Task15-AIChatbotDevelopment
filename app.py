import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )

    return tokenizer, model


tokenizer, model = load_model()


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer the user's question clearly and accurately. "
    "Use simple language. "
    "Keep answers concise, preferably 2 to 5 sentences. "
    "Do not invent facts. "
    "If you are unsure about something, say that you are not sure."
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


# ==========================================
# TITLE
# ==========================================

st.title("🤖 AI Chatbot")

st.write(
    "A simple AI chatbot powered by Qwen2.5-0.5B-Instruct."
)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("**Model:**")
    st.info("Qwen2.5-0.5B-Instruct")

    st.write("**Device:**")
    st.info("CPU")

    st.divider()

    st.write("### 💡 Example Questions")

    st.write("• What is Python?")
    st.write("• What is Machine Learning?")
    st.write("• What is Artificial Intelligence?")
    st.write("• What is Deep Learning?")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==========================================
# GENERATE RESPONSE
# ==========================================

def generate_response(user_input):

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Keep only recent conversation
    system_message = st.session_state.messages[0]

    recent_messages = st.session_state.messages[-10:]

    messages = [
        system_message
    ] + recent_messages

    # Create chat prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    # Generate
    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # Remove prompt tokens
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    # Decode
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    response = response.strip()

    # Save response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    return response


# ==========================================
# CHAT INPUT
# ==========================================

user_input = st.chat_input(
    "Ask me something..."
)


if user_input:

    with st.chat_message("user"):

        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            response = generate_response(
                user_input
            )

        st.write(response)