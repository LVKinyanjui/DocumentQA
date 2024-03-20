import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

def generate_content(prompt):
    return model.generate_content(prompt).text

if __name__ == "__main__":
    print(generate_content("What is life?"))