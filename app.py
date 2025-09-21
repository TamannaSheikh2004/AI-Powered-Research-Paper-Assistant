import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import openai  # DeepSeek uses OpenAI-compatible API

# --------------------
# API Configuration
# --------------------
load_dotenv()

# Commented Gemini for now
# import google.generativeai as genai
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure DeepSeek (OpenAI compatible)
openai.api_key = "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"
openai.api_base = "https://api.deepseek.com/v1"


def generate_ai_content(prompt):
    """Generate content using DeepSeek API"""
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Error generating content: {str(e)}]"


# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📑 AI Research Assistant")
st.write("Generate a research paper with custom formats, templates, resources, images, and graphs.")

# Step 1: Topic
topic = st.text_input("Enter your research topic")

# Step 2: Format
st.subheader("Research Paper Format")
use_format = st.radio("Do you have a research format?", ["Yes", "No"])
format_structure = None

if use_format == "Yes":
    format_file = st.file_uploader("Upload your format file (PDF)", type=["pdf"])
    if format_file is not None:
        pdf_reader = PyPDF2.PdfReader(format_file)
        format_structure = ""
        for page in pdf_reader.pages:
            format_structure += page.extract_text()
else:
    format_structure = """
    Abstract
    Introduction
    Literature Review
    Methodology
    Results
    Discussion
    Conclusion
    References
    """

# Step 3: Template
st.subheader("Research Paper Template")
use_template = st.radio("Do you have a research template?", ["Yes", "No"])
template_file = None
template_choice = None

if use_template == "Yes":
    template_file = st.file_uploader("Upload your template file (DOCX)", type=["docx"])
else:
    template_choice = st.selectbox(
        "Select a template",
        ["IEEE Style", "APA Style", "MLA Style", "Simple Academic"]
    )
    st.info(f"You selected: {template_choice}. A default styled template will be generated.")

# Step 4: Resources
st.subheader("Additional Resources")
resource_choice = st.radio("Do you have research resources to insert?", ["Yes", "No", "Autogenerate"])
user_resources = None
if resource_choice == "Yes":
    user_resources = st.file_uploader("Upload your resources (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
elif resource_choice == "Autogenerate":
    st.info("We will automatically generate related resources using AI.")

# Step 5: Images
st.subheader("Images")
image_choice = st.radio("Do you want to insert images?", ["Upload", "Autogenerate", "None"])
user_images = None
if image_choice == "Upload":
    user_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
elif image_choice == "Autogenerate":
    image_description = st.text_input("Describe the type of images you want (e.g., AI in daily life, Neural Networks diagram)")

# Step 6: Graphs
st.subheader("Graphs")
graph_choice = st.radio("Do you want to insert graphs?", ["Upload", "Autogenerate", "None"])
user_graphs = None
if graph_choice == "Upload":
    user_graphs = st.file_uploader("Upload graph images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
elif graph_choice == "Autogenerate":
    graph_description = st.text_input("Describe the graph you want (e.g., AI growth trend, accuracy comparison)")

# Step 7: Final Output Image
st.subheader("Final Output Image")
output_img_choice = st.radio("Do you want to add a final output image?", ["Upload", "Autogenerate", "None"])
final_output_image = None
if output_img_choice == "Upload":
    final_output_image = st.file_uploader("Upload final output image", type=["png", "jpg", "jpeg"])
elif output_img_choice == "Autogenerate":
    final_img_description = st.text_input("Describe the final image you want (e.g., futuristic AI concept art)")

# Step 8: Generate Paper
if st.button("Generate Research Paper"):
    if not topic:
        st.error("Please enter a research topic.")
    else:
        st.info("Generating research paper... ⏳ This may take a while.")

        # Sections
        sections = {
            "Abstract": f"Write an academic abstract about {topic}.",
            "Introduction": f"Write an introduction for a research paper on {topic}.",
            "Literature Review": f"Write a literature review for {topic}.",
            "Methodology": f"Write the methodology for a research paper on {topic}.",
            "Results": f"Discuss possible results/findings related to {topic}.",
            "Discussion": f"Write the discussion section for {topic}.",
            "Conclusion": f"Write a conclusion for {topic}.",
            "References": f"Provide sample references (APA style) for {topic}.",
        }

        paper_sections = {}
        for key, prompt in sections.items():
            paper_sections[key] = generate_ai_content(prompt)

        # Load or create template
        if template_file:
            doc = Document(template_file)
        else:
            doc = Document()
            doc.add_paragraph(f"{template_choice} Research Paper Template\n")

        # Insert generated content into the template safely
        existing_text = "\n".join([p.text for p in doc.paragraphs]).lower()

        for h, content in paper_sections.items():
            if h.lower() not in existing_text:
                try:
                    doc.add_heading(h, level=1)
                except KeyError:
                    p = doc.add_paragraph()
                    run = p.add_run(h)
                    run.bold = True
                for block in [b for b in content.split("\n\n") if b.strip()]:
                    doc.add_paragraph(block)

        # Save DOCX
        output_docx = "Generated_Research_Paper.docx"
        doc.save(output_docx)

        # Save PDF (basic fallback)
        output_pdf = "Generated_Research_Paper.pdf"
        pdf = canvas.Canvas(output_pdf, pagesize=letter)
        text_obj = pdf.beginText(40, 750)
        text_obj.setFont("Helvetica", 10)

        for section, content in paper_sections.items():
            text_obj.textLine(section)
            for line in content.split("\n"):
                text_obj.textLine(line)
            text_obj.textLine(" ")

        pdf.drawText(text_obj)
        pdf.save()

        # Download buttons
        with open(output_docx, "rb") as f:
            st.download_button(
                "📥 Download Research Paper (DOCX)",
                f,
                file_name="Generated_Research_Paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        with open(output_pdf, "rb") as f:
            st.download_button(
                "📥 Download Research Paper (PDF)",
                f,
                file_name="Generated_Research_Paper.pdf",
                mime="application/pdf"
            )

        st.success("Research paper generated successfully! 🎉")
