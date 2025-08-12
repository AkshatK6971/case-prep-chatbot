from typing import List, Dict
from typing_extensions import TypedDict
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_key = os.getenv("GROQ_KEY")

# ---------------- Neural Search ---------------- #
class Neural_Search:
    def __init__(self, collection_name="case_prep"):
        self.collection_name = collection_name
        self.model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.qdrant_client = QdrantClient("http://localhost:6333")

    def search(self, text: str):
        vector = self.model.embed_query(text)
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None,
            limit=5,
        ).points
        payloads = [hit.payload for hit in search_result]
        return payloads


# ---------------- Conversation State ---------------- #
class ConversationState(TypedDict):
    history_summary: str
    messages: List[Dict[str, str]]
    intent: str
    context_docs: List[str]
    response: str


# ---------------- Shared Objects ---------------- #
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, api_key=groq_key)
parser = StrOutputParser()
searcher = Neural_Search()

# ---------------- Intent Classification ---------------- #
intent_prompt = ChatPromptTemplate.from_template("""
System: You are an expert in consulting case interview assistance and conversation intent detection.
Your job is to read the ENTIRE conversation so far and classify the user's LATEST message into one of the predefined intent labels.

Instructions:
- Carefully analyze the conversation history and the latest user message in context.
- Consider both explicit requests and implied intent.
- Output **only one exact label** from the list below — no explanations, extra words, or formatting.
- If multiple intents seem relevant, choose the one that MOST directly reflects the immediate purpose of the latest user query.

Possible intent labels:
- GENERATE_CASE: The user requests you to create a new case interview question or problem.
- ANSWER_CASE: The user is asking you to solve, analyze, or answer a specific case interview question.
- ASK_FRAMEWORK: The user is requesting a case-solving framework, structure, or approach.
- HELP: The user is asking for guidance on how to use the assistant or interact with it.
- GENERAL_QUERY: The user is asking a question unrelated to case interviews or frameworks.
- FEEDBACK: The user is requesting evaluation or comments on their case interview performance or response.

Conversation so far:
{history_summary}

Latest user message:
{query}

Output:
(Only the exact label from the list above.)
""")

def classify_intent(state: ConversationState):
    result = intent_prompt | llm | parser
    label = result.invoke({"history_summary": state["history_summary"], "query": state["messages"][-1]["content"]})
    state["intent"] = label.strip()
    return state


# ---------------- Context Retrieval ---------------- #
def retrieve_context(state: ConversationState):
    query = state["messages"][-1]["content"]
    docs = searcher.search(query)
    state["context_docs"] = [doc.get("text", "") for doc in docs]
    return state


# ---------------- Prompts ---------------- #
case_prompt = ChatPromptTemplate.from_template("""
System: You are a highly skilled consulting case interview preparation assistant.
Your role is to provide clear, structured, and insightful responses to help the user improve their case-solving skills.

Instructions:
- Review the conversation summary and provided context carefully.
- Use relevant consulting principles, frameworks, and logical reasoning.
- If the user’s query is unclear, briefly ask clarifying questions before answering.
- Provide your answer in a structured manner (e.g., bullet points, MECE principles, steps).
- Maintain a professional, constructive, and educational tone.

Conversation Summary:
{history_summary}

Context:
{context}

User query:
{query}

Now provide the best possible answer.
""")

general_prompt = ChatPromptTemplate.from_template("""
System: You are a friendly, knowledgeable, and concise AI assistant.
You respond helpfully and clearly, while adapting your tone to be approachable yet professional.

Instructions:
- Read the conversation summary to understand context.
- Directly answer the user’s question, avoiding unnecessary repetition.
- Keep answers concise but complete — use bullet points or short paragraphs if helpful.
- If relevant, provide examples, analogies, or references for better understanding.
- Avoid speculative or inaccurate information — if unsure, acknowledge it.

Conversation Summary:
{history_summary}

User query:
{query}

Now provide the clearest and most helpful response possible.
""")

feedback_prompt = ChatPromptTemplate.from_template("""
System: You are an expert case interview coach.
Your goal is to give **constructive, actionable, and motivating** feedback on the user's case interview answer or performance.

Instructions:
1. Read the user's answer carefully for content, clarity, structure, and problem-solving logic.
2. In your feedback, ALWAYS include:
    a) **Strengths** — specific examples of what the user did well (e.g., strong structuring, insight, numerical accuracy).
    b) **Areas for Improvement** — pinpoint missing elements, unclear logic, or consulting best practices they didn't apply.
    c) **Actionable Suggestions** — concrete steps to improve responses (e.g., use MECE structures, prioritize issues, improve quant analysis).
    d) **Relevant Framework Guidance** (if applicable) — mention which frameworks or approaches could have improved their answer.
3. Maintain a professional, supportive tone that builds confidence while guiding improvement.
4. Be concise but detailed enough for the user to understand exactly what to do better next time.

Conversation Summary:
{history_summary}

User's case answer or performance description:
{query}

Now provide your structured, motivating feedback.
""")

help_prompt = ChatPromptTemplate.from_template("""
System: You are a friendly and knowledgeable onboarding guide for this case interview preparation chatbot. 
Your job is to clearly explain to a new user:
1. The types of queries they can ask.
2. How the assistant uses Retrieval-Augmented Generation (RAG) with Qdrant for case-specific answers.
3. Example queries for each type.
4. Tips for getting the best results.

Instructions:
- Use a warm, encouraging, and approachable tone while remaining professional.
- Organize your answer into **clear, well-labeled sections**:
    a) "What You Can Ask" — describe and briefly explain the five main types:
       • Generate Cases
       • Answer Cases  
       • Get Frameworks  
       • Ask General Questions  
       • Request Feedback
    b) "How It Works" — explain how the chatbot uses RAG with Qdrant to find and integrate relevant case data for more accurate answers.
    c) "Example Commands" — 1-2 concrete query examples for each type.
    d) "Pro Tips" — 3-5 practical tips on phrasing questions, requesting clarifications, and using structured inputs for better results.
- Keep explanations **concise** but **informative**.
- Avoid jargon without explanation — explain RAG and Qdrant in simple terms.
- Where possible, use bullet points for readability.

Conversation Summary:
{history_summary}

User's help request:
{query}

Now provide the complete, friendly onboarding guide.
""")

# ---------------- Response Generation ---------------- #
def generate_response(state: ConversationState):
    query = state["messages"][-1]["content"]

    if state["intent"] in ["GENERATE_CASE", "ANSWER_CASE", "ASK_FRAMEWORK"]:
        prompt = case_prompt
        ctx = "\n".join(state["context_docs"])

    elif state["intent"] == "GENERAL_QUERY":
        prompt = general_prompt
        ctx = ""

    elif state["intent"] == "FEEDBACK":
        prompt = feedback_prompt
        ctx = ""

    elif state["intent"] == "HELP":
        prompt = help_prompt
        ctx = ""

    else:
        prompt = general_prompt
        ctx = ""

    chain = prompt | llm | parser
    state["response"] = chain.invoke({"history_summary": state["history_summary"], "context": ctx, "query": query})
    return state


# ---------------- Summary Update ---------------- #
summary_prompt = ChatPromptTemplate.from_template("""
System: You are an expert summarizer for multi-turn chatbot conversations.
Your job is to produce a **clear, concise, and complete** conversation summary that can be used as memory for future interactions.

Instructions:
- Incorporate the **entire conversation so far**, not just the latest turn.
- Capture key details, decisions, clarifications, user preferences, and examples.
- Preserve all **technical terms, case interview frameworks, numbers, or scenarios** mentioned.
- Always include the **latest user query** and the assistant’s most recent response verbatim or very close in meaning.
- Keep the summary factual, neutral, and chronological.
- Avoid unnecessary commentary or personal opinions.

Previous conversation context:
{history_summary}

Latest user message(s):
{new_messages}

Now produce the updated conversation summary.
""")

def update_history_summary(state: ConversationState):
    new_messages_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in state["messages"][-2:]]
    )
    chain = summary_prompt | llm | parser
    new_summary = chain.invoke({"history_summary": state["history_summary"], "new_messages": new_messages_text})
    state["history_summary"] = new_summary
    return state


# ---------------- LangGraph Workflow ---------------- #
workflow = StateGraph(ConversationState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_response", generate_response)
workflow.add_node("update_summary", update_history_summary)

workflow.add_edge("classify_intent", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", "update_summary")

workflow.set_entry_point("classify_intent")
app = workflow.compile()


# ---------------- Run Loop ---------------- #
# if __name__ == "__main__":
#     state: ConversationState = {
#         "history_summary": "",
#         "messages": [],
#         "intent": "",
#         "context_docs": [],
#         "response": ""
#     }

#     while True:
#         user_input = input("[YOU]: ")
#         if user_input.lower() in ["quit", "exit"]:
#             break
#         state["messages"].append({"role": "user", "content": user_input})
#         state = app.invoke(state)
#         print(f"[ASSISTANT]: {state['response']}")
#         state["messages"].append({"role": "assistant", "content": state['response']})