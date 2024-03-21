import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def generate_content(prompt):
    try:
        res = model.generate_content(prompt).text

    # Handle errors that result from google blocking the response.
    except ValueError:
        res = "Could not answer your question. Please try another one."

    return res


if __name__ == "__main__":
    print(generate_content("What is life?"))