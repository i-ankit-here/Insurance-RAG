import os
from dotenv import load_dotenv

load_dotenv()

from google import genai
import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize Gemini SDK
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")
client = genai.Client(api_key=google_key)

# 2. CRITICAL: Match the database's embedding model!
bge_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)

# 3. Connect to your Local Database
db_client = chromadb.PersistentClient(path="./insurance_db")
collection = db_client.get_collection(
    name="health_policies",
    embedding_function=bge_embeddings
)

def insurance_agent(user_query, session_policy=None):
    # PHASE 0: INTENT ROUTER
    intent_prompt = f"Categorize this user query into one of two categories: 'RECOMMENDATION' or 'CLAIM_CHECK'. Query: '{user_query}'. Return ONLY the category name."
    intent = client.models.generate_content(model="gemini-2.5-flash", contents=intent_prompt).text.strip()

    # PATH A: THE RECOMMENDATION ENGINE
    if "RECOMMENDATION" in intent:
        # Search using the BGE-Large model automatically
        results = collection.query(query_texts=[user_query], n_results=5)
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No matching features found."

        rec_prompt = f"""
        You are an expert Insurance Broker. 
        USER REQUEST: {user_query}
        AVAILABLE POLICY DATA: {context}
        
        TASK: Recommend the best policy from the data that fits the user's needs. 
        Explain WHY based on the features.
        """
        response = client.models.generate_content(model="gemini-2.5-flash", contents=rec_prompt)
        return response.text, session_policy

    # PATH B: THE CLAIM JUDGE
    else:
        # Phase 1: Policy Detection 
        if not session_policy:
            detect_prompt = f"Extract the insurance policy name from this query: '{user_query}'. If no specific brand/policy is mentioned, return 'UNKNOWN'."
            detection = client.models.generate_content(model="gemini-2.5-flash", contents=detect_prompt).text.strip()
            
            if "UNKNOWN" in detection:
                return "I'd be happy to check that claim! **Which specific health insurance policy do you have?**", None
            session_policy = detection.lower()

        # Phase 2: Targeted Retrieval
        results = collection.query(
            query_texts=[user_query],
            n_results=5,
            where={"policy_name": {"$contains": session_policy}} 
        )
        context = "\n\n".join(results['documents'][0]) if results['documents'] else "No specific policy text found."

        # Phase 3: Reasoning
        reasoning_prompt = f"""
        You are an expert Health Insurance Claims Judge.
        USER POLICY CONTEXT: {context}
        USER QUESTION: {user_query}

        TASK:
        1. Judge if the claim is possible based ONLY on the provided context.
        2. Cite the specific Clause or Page if mentioned in the text.
        3. If the context does not contain the answer, state that clearly.
        """
        response = client.models.generate_content(model="gemini-2.5-flash", contents=reasoning_prompt)
        return response.text, session_policy

# TERMINAL CHAT LOOP 
if __name__ == "__main__":
    for m in client.models.list():
      print(f"Model Name: {m.name}")
    current_policy = None
    print("AI Insurance Broker & Claim Judge")
    
    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ['exit', 'quit']:
            break
            
        ans, current_policy = insurance_agent(u_input, current_policy)
        print(f"\nAI: {ans}")