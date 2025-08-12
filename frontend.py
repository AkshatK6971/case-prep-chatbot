import streamlit as st
from case_prep import app

st.set_page_config(page_title="Case Interview Assistant", page_icon="💼")
st.title("💼 Case Interview Preparation Assistant")

# --- Initialize session state --- #
if "state" not in st.session_state:
    st.session_state.state = {
        "history_summary": "",
        "messages": [],
        "intent": "",
        "context_docs": [],
        "response": ""
    }

# --- Clear chat --- #
if st.button("🗑️ Clear Chat"):
    st.session_state.state = {
        "history_summary": "",
        "messages": [],
        "intent": "",
        "context_docs": [],
        "response": ""
    }
    st.rerun()

# --- Display messages --- #
for msg in st.session_state.state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# --- Chat input --- #
if prompt := st.chat_input("Type your message..."):
    st.session_state.state["messages"].append({"role": "user", "content": prompt})
    st.session_state.state = app.invoke(st.session_state.state)
    st.session_state.state["messages"].append({"role": "assistant", "content": st.session_state.state["response"]})

    st.rerun()
