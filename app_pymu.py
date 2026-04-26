import os
from dotenv import load_dotenv

load_dotenv()

from google import genai
import chromadb

# Initialize the Gemini SDK and Local Database
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file!")

client = genai.Client(api_key=google_key)
db_client = chromadb.PersistentClient(path="./insurance_db")
collection = db_client.get_collection("health_policies")

def insurance_agent(user_query, session_policy=None):
    # PHASE 0: INTENT ROUTER
    intent_prompt = f"Categorize this user query into one of two categories: 'RECOMMENDATION' or 'CLAIM_CHECK'. Query: '{user_query}'. Return ONLY the category name."
    intent_response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=intent_prompt
    )
    intent = intent_response.text.strip()
    
    # PATH A: THE RECOMMENDATION ENGINE
    if "RECOMMENDATION" in intent:
        # Search the database locally based on the user's features
        results = collection.query(
            query_texts=[user_query], 
            n_results=5
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            context = "\n\n".join(results['documents'][0])
        else:
            context = "No matching features found in the database."

        rec_prompt = f"""
        You are an expert Insurance Broker. 
        USER REQUEST: {user_query}
        AVAILABLE POLICY DATA: {context}
        
        TASK: Recommend the best policy/policies from the data that fit the user's needs. 
        Explain WHY based on the features. If the data doesn't contain a good match, clearly state that you cannot find a suitable policy in the current documents.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=rec_prompt
        )
        return response.text, session_policy

    # PATH B: THE CLAIM JUDGE
    else:
        # PHASE 1: POLICY DETECTION
        if not session_policy:
            detect_prompt = f"Extract the insurance policy name from this query: '{user_query}'. If no specific insurance brand or policy name is mentioned, return exactly 'UNKNOWN'."
            
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=detect_prompt
            )
            detection = response.text.strip()
            
            if "UNKNOWN" in detection:
                return "I'd be happy to check that claim for you! **Which specific health insurance policy do you have?**", None
            session_policy = detection.lower()

        # PHASE 2: RETRIEVAL
        results = collection.query(
            query_texts=[user_query],
            n_results=5,
            where={"policy_name": {"$contains": session_policy}}
        )
        
        if results['documents'] and len(results['documents'][0]) > 0:
            context = "\n\n".join(results['documents'][0])
        else:
            context = "No specific policy text found."

        # PHASE 3: REASONING
        reasoning_prompt = f"""
        You are an expert Health Insurance Claims Judge.
        USER POLICY CONTEXT: {context}
        USER QUESTION: {user_query}

        TASK:
        1. Judge if the claim is possible based ONLY on the provided context.
        2. Cite the specific Clause or Page if mentioned in the text.
        3. If the context does not contain the answer, state that clearly.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=reasoning_prompt
        )
        return response.text, session_policy

# TERMINAL CHAT LOOP 
if __name__ == "__main__":
    current_policy = None
    print("AI Insurance Broker & Claim Judge Prototype")
    print("Type 'exit' to quit.\n")
    
    while True:
        u_input = input("\nYou: ")
        if u_input.lower() in ['exit', 'quit']:
            break
            
        ans, current_policy = insurance_agent(u_input, current_policy)
        print(f"\nAI: {ans}")