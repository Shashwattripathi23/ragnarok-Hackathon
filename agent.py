import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client securely
# Make sure GROQ_API_KEY is in your .env file!
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found! Please create a .env file.")

client = Groq(api_key=api_key)

def ask_study_agent(user_query, context=""):
    """
    Sends a prompt to the Groq API. 
    """

    system_prompt = (
        "You are an highly intelligent AI Study Assistant. "
        "Your goal is to help a student learn effectively based on the provided course material context. "
        "Do not give direct answers to graded assignments; instead, explain the concepts."
    )
    
    if context:
        system_prompt += f"\n\nCOURSE MATERIAL CONTEXT:\n{context}"

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5, 
            max_tokens=1024,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Agent Error: {str(e)}"