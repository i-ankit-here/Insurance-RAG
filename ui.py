import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(page_title="AI Insurance Broker", page_icon="")
st.title("AI Insurance Broker and Claim Judge")

@st.cache_resource
def load_system():
    load_dotenv()
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        st.error("Missing GOOGLE_API_KEY in .env file")
        st.stop()
        
    client = genai.Client(api_key=google_key)
    
    bge_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-large-en-v1.5"
    )
    
    db_client = chromadb.PersistentClient(path="./insurance_db")
    collection = db_client.get_collection(
        name="health_policies",
        embedding_function=bge_embeddings
    )
    return client, collection

client, collection = load_system()

def insurance_agent(user_query, chat_history_str, session_policy=None):

    rewrite_prompt = f"""
    CONVERSATION HISTORY: 
    {chat_history_str}
    NEW USER MESSAGE: '{user_query}'
    TASK: Rewrite the NEW USER MESSAGE into a concise, standalone ChromaDB search query. 
    RULES:
    1. Resolve all pronouns to the specific policy name from the history.
    2. Expand insurance acronyms.
    3. Remove any conversational fluff or context that is not directly relevant to the search intent.
    4. Focus on extracting the core information need that can be used to query the policy database effectively.
    5. If the query is ambiguous, prioritize extracting the most specific policy-related terms.
    6. Use the conversation history to infer any missing details that would make the search query more effective, but do not include irrelevant information.
    6. Use the conversation history to infer any missing details that would make the search query more effective, but do not include irrelevant information.
    7. Ensure the final search query is optimized for retrieving relevant policy features or clauses from the database.
    8. Return ONLY the standalone search query, nothing else.
    """
    standalone_query = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=rewrite_prompt).text.strip()
    
    intent_prompt = f"Categorize this query into 'COMPARISON', 'SCENARIO', 'RECOMMENDATION', or 'CLAIM_CHECK'. Query: '{standalone_query}'. Return ONLY the category name. Be careful to choose the most relevant category based on the user's intent. If the query is primarily about comparing policies, return 'COMPARISON'. If it's about calculating costs based on a scenario, return 'SCENARIO'. If it's about asking for advice or recommendations, return 'RECOMMENDATION'. If it's about checking if a claim would be covered, return 'CLAIM_CHECK'."

    intent = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=intent_prompt).text.strip()

    if "COMPARISON" in intent:
        results = collection.query(query_texts=[standalone_query], n_results=100)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No matching features found."

        comp_prompt = f"""
        You are an expert Insurance Broker. 
        CONVERSATION HISTORY: {chat_history_str}
        USER REQUEST: {user_query}
        AVAILABLE POLICY DATA: {context}
        
        TASK: Compare the policies based on the request using the available data. Provide a Markdown table detailing the differences.
        """
        response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=comp_prompt)
        return response.text, session_policy

    elif "SCENARIO" in intent:
        results = collection.query(query_texts=[standalone_query], n_results=100)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No matching features found."

        scen_prompt = f"""
        You are a claims calculator. 
        CONVERSATION HISTORY: {chat_history_str}
        USER SCENARIO: {user_query}
        POLICY CONTEXT: {context}
        
        TASK: Step-by-step, calculate the estimated out-of-pocket expenses based on the context provided.
        """
        response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=scen_prompt)
        return response.text, session_policy

    elif "RECOMMENDATION" in intent:
        results = collection.query(query_texts=[standalone_query], n_results=100)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No matching features found."

        rec_prompt = f"""
        You are an expert Insurance Broker. 
        CONVERSATION HISTORY: {chat_history_str}
        USER REQUEST: {user_query}
        AVAILABLE POLICY DATA: {context}
        
        TASK: Answer the user's request using the available policy data. Keep the tone conversational and acknowledge the history if necessary. 
        You need to provide recommendations based on the user's request and the policy data. 
        If the user is asking for advice, give a clear recommendation. 
        You can recommend more than one policy if appropriate and ask the user if they want to compare them. 
        If the user is asking for information, provide that information in a helpful way. 
        Always use the context to inform your response, but do not include irrelevant details.
        """
        response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=rec_prompt)
        return response.text, session_policy
    
    elif "CLAIM_CHECK" in intent:
        if not session_policy:
            detect_prompt = f"Extract the insurance policy name from this query: '{standalone_query}'. Return 'UNKNOWN' if none."
            detection = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=detect_prompt).text.strip()
            
            if "UNKNOWN" in detection:
                return "I would be happy to check that. Which specific health insurance policy do you have?", None
            session_policy = detection.lower()

        results = collection.query(
            query_texts=[standalone_query],
            n_results=100,
            where={"policy_name": {"$contains": session_policy}} 
        )
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No specific policy text found."

        reasoning_prompt = f"""
        You are an expert Health Insurance Claims Judge.
        CONVERSATION HISTORY: {chat_history_str}
        USER POLICY CONTEXT: {context}
        USER QUESTION: {user_query}

        TASK: Answer the user's question based ONLY on the context. If the context does not contain the answer, state that clearly.
        """
        response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=reasoning_prompt)
        return response.text, session_policy
    else:
        results = collection.query(query_texts=[standalone_query], n_results=100)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No matching features found."

        rec_prompt = f"""
        You are an expert Insurance Advisor. 
        CONVERSATION HISTORY: {chat_history_str}
        USER REQUEST: {user_query}
        AVAILABLE POLICY DATA: {context}
        
        TASK: Answer the user's request using the available context, documents and user query. Keep the tone conversational and acknowledge the history if necessary. 
        If the context does not contain the answer, respond him that the information is not available and respond accordingly.
        """
        response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=rec_prompt)
        return response.text, session_policy
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help with your insurance today?"}]
if "current_policy" not in st.session_state:
    st.session_state.current_policy = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a policy, compare plans, or run a scenario"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    history_str = ""
    for msg in st.session_state.messages[-4:]:
        history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents"):
            ai_response, updated_policy = insurance_agent(prompt, history_str, st.session_state.current_policy)
            st.session_state.current_policy = updated_policy
            st.markdown(ai_response)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_response})