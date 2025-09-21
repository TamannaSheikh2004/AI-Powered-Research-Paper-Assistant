import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Ask user for input
topic = input("Enter your research topic: ")
keywords = input("Enter keywords (comma separated): ")

use_pdf = input("Do you have a PDF format template? (yes/no): ").strip().lower()

pdf_structure = ""
if use_pdf == "yes":
    pdf_path = input("Enter path to your PDF template (e.g., C:/Users/admin/Desktop/template.pdf): ")
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                pdf_structure += page.extract_text() + "\n"
        print("✅ PDF template loaded successfully!")
    except Exception as e:
        print("⚠️ Error reading PDF:", e)
        pdf_structure = "Use a simple academic research paper structure."
else:
    pdf_structure = "Use a standard academic format with Abstract, Introduction, Literature Review, Methodology, Conclusion, References."

# Create prompt
prompt = f"""
You are an AI-powered Research Paper Assistant.
Generate a research paper draft on the topic: {topic}.
Keywords: {keywords}.
Follow this format structure:
{pdf_structure}
"""

# Use Gemini
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)

# Print result
print("\n===== AI-Generated Research Paper Draft =====\n")
print(response.text)
