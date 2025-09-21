# app.py
import os
import io
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph
from docx.shared import Inches
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import PyPDF2

# ----------------------------
# Configuration
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Put key in .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")  # adjust if needed

if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY not found in .env — generation will fail until you set it.")

# ----------------------------
# Helpers: call Groq/OpenAI-compatible endpoint via requests
# ----------------------------
def generate_text_groq(prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint and return text."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing (set it in .env).")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful academic research assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    j = r.json()
    # handle common response shapes
    if "choices" in j and len(j["choices"]) > 0:
        c = j["choices"][0]
        # chat-style
        if isinstance(c.get("message"), dict) and "content" in c["message"]:
            return c["message"]["content"].strip()
        # older style
        if "text" in c:
            return c["text"].strip()
    # fallback: try 'output' style
    if "output" in j:
        text_acc = []
        for item in j["output"]:
            if isinstance(item, dict) and "content" in item:
                for c in item["content"]:
                    if c.get("type") == "output_text":
                        text_acc.append(c.get("text", ""))
        return "\n".join(text_acc).strip()
    return ""

# ----------------------------
# docx helpers
# ----------------------------
def insert_paragraph_after(paragraph: Paragraph, text: str = "", style=None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        try:
            new_para.style = style
        except Exception:
            pass
    if text:
        new_para.add_run(text)
    return new_para

def delete_paragraph(paragraph: Paragraph):
    p = paragraph._element
    parent = p.getparent()
    if parent is not None:
        parent.remove(p)

# ----------------------------
# simple image / graph generators (placeholders)
# ----------------------------
def generate_placeholder_image(text: str, width=800, height=450) -> io.BytesIO:
    img = Image.new("RGB", (width, height), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        fnt = ImageFont.load_default()
    # naive wrapping
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur + " " + w) > 40:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    y = 30
    for ln in lines[:8]:
        draw.text((20, y), ln, font=fnt, fill=(20,20,20))
        y += 28
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def generate_sample_graph(kind="bar") -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6,3))
    if kind == "bar":
        ax.bar(["A","B","C"], [3,7,5])
    elif kind == "line":
        ax.plot([1,2,3,4],[1,3,2,5])
    elif kind == "pie":
        ax.pie([30,50,20], labels=["X","Y","Z"], autopct="%1.0f%%")
    ax.set_title(kind.capitalize()+" example")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return buf

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Research Assistant — DeepSeek", layout="wide")
st.title("AI Research Assistant — DeepSeek (Groq)")

# 1) Topic
topic = st.text_input("1) Research topic", "")

# 2) Format / headings
st.subheader("2) Format / Structure")
have_format = st.radio("Do you have a format (list of headings)?", ("No","Yes"))
user_headings = None
if have_format == "Yes":
    pasted = st.text_area("Paste headings (one per line)", height=120)
    filefmt = st.file_uploader("Optional: upload a PDF that lists your headings", type=["pdf"])
    if filefmt:
        try:
            reader = PyPDF2.PdfReader(filefmt)
            txt = ""
            for p in reader.pages:
                txt += (p.extract_text() or "") + "\n"
            user_headings = [line.strip() for line in txt.splitlines() if line.strip()]
        except Exception:
            user_headings = [line.strip() for line in pasted.splitlines() if line.strip()]
    else:
        user_headings = [line.strip() for line in pasted.splitlines() if line.strip()]

if not user_headings:
    user_headings = ["Abstract","Introduction","Literature Review","Methodology","Results","Discussion","Conclusion","References"]

# 3) Template
st.subheader("3) Template")
have_template = st.radio("Do you have a DOCX template?", ("Yes","No"))
uploaded_template = None
template_choice = None
if have_template == "Yes":
    uploaded_template = st.file_uploader("Upload DOCX template (will insert text below headings)", type=["docx"])
else:
    template_choice = st.selectbox("Pick a default template style", ["Simple Academic","IEEE","APA","MLA"])
    st.info("A simple DOCX template will be generated if you don't upload one.")

# 4) Resources
st.subheader("4) Resources")
res_choice = st.radio("Provide resources?", ("No","Upload","Autogenerate"))
uploaded_resources = []
resources_description = ""
if res_choice == "Upload":
    uploaded_resources = st.file_uploader("Upload PDF/DOCX/TXT", type=["pdf","docx","txt"], accept_multiple_files=True)
elif res_choice == "Autogenerate":
    resources_description = st.text_area("Describe what resources to autogenerate (short)")

# 5) Images
st.subheader("5) Images")
img_choice = st.radio("Images", ("None","Upload","Autogenerate"))
uploaded_images = []
img_description = ""
if img_choice == "Upload":
    uploaded_images = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
elif img_choice == "Autogenerate":
    img_description = st.text_input("Describe image(s) to generate")

# 6) Graphs
st.subheader("6) Graphs")
graph_choice = st.radio("Graphs", ("None","Upload","Autogenerate"))
uploaded_graphs = []
graph_description = ""
if graph_choice == "Upload":
    uploaded_graphs = st.file_uploader("Upload graph images", type=["png","jpg","jpeg"], accept_multiple_files=True)
elif graph_choice == "Autogenerate":
    graph_description = st.text_input("Describe graph to generate (bar/line/pie)")

# 7) Final image
st.subheader("7) Final output image")
final_choice = st.radio("Final image", ("None","Upload","Autogenerate"))
uploaded_final_image = None
final_description = ""
if final_choice == "Upload":
    uploaded_final_image = st.file_uploader("Upload final image", type=["png","jpg","jpeg"])
elif final_choice == "Autogenerate":
    final_description = st.text_input("Describe final image to generate")

# Generate button
if st.button("Generate research paper"):
    if not topic:
        st.error("Type a topic first.")
    else:
        st.info("Generating content... this can take a minute.")

        # Build prompts for headings
        prompts = {}
        for h in user_headings:
            low = h.lower()
            if "abstract" in low:
                prompts[h] = f"Write a concise academic abstract for a paper on: {topic}"
            elif "introduc" in low:
                prompts[h] = f"Write an introduction for a research paper on: {topic}"
            elif "literature" in low or "related" in low:
                prompts[h] = f"Write a short literature review summarizing common approaches and gaps for: {topic}"
            elif "method" in low:
                prompts[h] = f"Write a methodology section sketch for research on {topic} including potential datasets/experiments"
            elif "result" in low:
                prompts[h] = f"Describe plausible results/findings and how they'd be presented for {topic}"
            elif "discuss" in low:
                prompts[h] = f"Write a discussion section reflecting implications and limitations for {topic}"
            elif "conclusion" in low:
                prompts[h] = f"Write a clear conclusion for {topic}"
            elif "refer" in low:
                prompts[h] = f"Provide 5 sample references (title, authors, year, venue) relevant to {topic}"
            else:
                prompts[h] = f"Write the section titled '{h}' for a research paper about {topic}"

        paper_sections = {}
        for h, p in prompts.items():
            try:
                paper_sections[h] = generate_text_groq(p, max_tokens=900)
            except Exception as e:
                paper_sections[h] = f"[Error generating {h}: {e}]"

        # Summarize uploaded resources if any
        resource_summaries = []
        if res_choice == "Upload" and uploaded_resources:
            for f in uploaded_resources:
                name = f.name
                text_chunk = ""
                try:
                    if name.lower().endswith(".pdf"):
                        reader = PyPDF2.PdfReader(f)
                        for pg in reader.pages:
                            text_chunk += (pg.extract_text() or "") + "\n"
                    elif name.lower().endswith(".docx"):
                        tmp = Document(f)
                        for p in tmp.paragraphs:
                            text_chunk += (p.text or "") + "\n"
                    else:
                        text_chunk = f.getvalue().decode(errors="ignore")
                except Exception:
                    text_chunk = ""
                if text_chunk:
                    short = text_chunk[:3000]
                    try:
                        summary = generate_text_groq(f"Summarize this resource briefly:\n\n{short}", max_tokens=400)
                    except Exception as e:
                        summary = f"[Error summarizing: {e}]"
                else:
                    summary = "[No text extracted]"
                resource_summaries.append((name, summary))
        elif res_choice == "Autogenerate" and resources_description:
            try:
                resource_text = generate_text_groq(f"Generate 2 short resource summaries relevant to {topic}: {resources_description}", max_tokens=400)
                resource_summaries.append(("AI-generated", resource_text))
            except Exception as e:
                resource_summaries.append(("AI-generated", f"[Error: {e}]"))

        # Prepare images/graphs buffers
        images_to_insert = []
        if img_choice == "Upload" and uploaded_images:
            for im in uploaded_images:
                images_to_insert.append(im)
        elif img_choice == "Autogenerate" and img_description:
            images_to_insert.append(generate_placeholder_image(img_description))

        graphs_to_insert = []
        if graph_choice == "Upload" and uploaded_graphs:
            for g in uploaded_graphs:
                graphs_to_insert.append(g)
        elif graph_choice == "Autogenerate" and graph_description:
            # choose graph type naive mapping
            kind = "bar"
            if "line" in graph_description.lower():
                kind = "line"
            elif "pie" in graph_description.lower():
                kind = "pie"
            graphs_to_insert.append(generate_sample_graph(kind=kind))

        final_image_buf = None
        if final_choice == "Upload" and uploaded_final_image:
            final_image_buf = uploaded_final_image
        elif final_choice == "Autogenerate" and final_description:
            final_image_buf = generate_placeholder_image(final_description)

        # Load template or create simple doc
        if uploaded_template:
            with open("uploaded_template.docx", "wb") as tmpf:
                tmpf.write(uploaded_template.getbuffer())
            doc = Document("uploaded_template.docx")
        else:
            doc = Document()
            doc.add_heading(topic, level=0)
            doc.add_paragraph(f"Generated using Groq/DeepSeek model: {GROQ_MODEL}")

        # Insert generated sections under matching headings (delete existing content under heading)
        paras = doc.paragraphs
        i = 0
        heading_keys = list(paper_sections.keys())
        while i < len(paras):
            para = paras[i]
            text = (para.text or "").strip()
            if not text:
                i += 1
                continue
            matched = None
            tnorm = text.lower()
            for h in heading_keys:
                hn = h.lower()
                if tnorm == hn or tnorm.startswith(hn) or hn in tnorm:
                    matched = h
                    break
            if matched:
                # remove following paragraphs until next heading
                j = i + 1
                while j < len(paras):
                    nt = (paras[j].text or "").strip().lower()
                    if not nt:
                        delete_paragraph(paras[j])
                        paras = doc.paragraphs
                        continue
                    is_next_heading = False
                    for h2 in heading_keys:
                        hn2 = h2.lower()
                        if nt == hn2 or nt.startswith(hn2) or hn2 in nt:
                            is_next_heading = True
                            break
                    if is_next_heading:
                        break
                    delete_paragraph(paras[j])
                    paras = doc.paragraphs
                # insert generated content
                insertion = para
                content = paper_sections.get(matched, "")
                blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
                if not blocks and content.strip():
                    blocks = [content.strip()]
                for block in blocks:
                    insertion = insert_paragraph_after(insertion, text=block, style=para.style)
                paras = doc.paragraphs
                i += len(blocks) + 1
            else:
                i += 1

        # Append any sections that weren't matched
        existing_text = " ".join([p.text for p in doc.paragraphs]).lower()
        for h, content in paper_sections.items():
            if h.lower() not in existing_text:
                try:
                    doc.add_heading(h, level=1)
                except Exception:
                    p = doc.add_paragraph()
                    r = p.add_run(h)
                    r.bold = True
                for block in [b for b in content.split("\n\n") if b.strip()]:
                    doc.add_paragraph(block)

        # Insert resource summaries (before References if present)
        refs_idx = None
        for idx, p in enumerate(doc.paragraphs):
            if "reference" in (p.text or "").lower():
                refs_idx = idx
                break
        if resource_summaries:
            if refs_idx:
                anchor = doc.paragraphs[refs_idx]
                ins = anchor
                for name, summ in resource_summaries:
                    ins = insert_paragraph_after(ins, text=f"Resource: {name}", style=anchor.style)
                    ins = insert_paragraph_after(ins, text=summ, style=anchor.style)
            else:
                doc.add_heading("Resources", level=1)
                for name, summ in resource_summaries:
                    doc.add_paragraph(f"{name}:")
                    doc.add_paragraph(summ)

        # Insert visuals near Results if present, else at end
        results_idx = None
        for idx, p in enumerate(doc.paragraphs):
            if "result" in (p.text or "").lower():
                results_idx = idx
                break
        # if results_idx None -> append at end
        if images_to_insert or graphs_to_insert or final_image_buf:
            if results_idx is not None:
                # insert after results heading
                anchor = doc.paragraphs[results_idx]
                # insert a heading for Figures
                ins = insert_paragraph_after(anchor, text="Figures", style=anchor.style)
                for im in images_to_insert:
                    if hasattr(im, "getbuffer"):
                        doc.add_picture(io.BytesIO(im.getbuffer()), width=Inches(5))
                    else:
                        doc.add_picture(im, width=Inches(5))
                if graphs_to_insert:
                    for gr in graphs_to_insert:
                        if hasattr(gr, "getbuffer"):
                            doc.add_picture(io.BytesIO(gr.getbuffer()), width=Inches(5))
                        else:
                            doc.add_picture(gr, width=Inches(5))
                if final_image_buf:
                    if hasattr(final_image_buf, "getbuffer"):
                        doc.add_paragraph("Final Image")
                        doc.add_picture(io.BytesIO(final_image_buf.getbuffer()), width=Inches(5))
                    else:
                        doc.add_picture(final_image_buf, width=Inches(5))
            else:
                doc.add_heading("Figures", level=2)
                for im in images_to_insert:
                    if hasattr(im, "getbuffer"):
                        doc.add_picture(io.BytesIO(im.getbuffer()), width=Inches(5))
                    else:
                        doc.add_picture(im, width=Inches(5))
                for gr in graphs_to_insert:
                    if hasattr(gr, "getbuffer"):
                        doc.add_picture(io.BytesIO(gr.getbuffer()), width=Inches(5))
                    else:
                        doc.add_picture(gr, width=Inches(5))
                if final_image_buf:
                    if hasattr(final_image_buf, "getbuffer"):
                        doc.add_paragraph("Final Image")
                        doc.add_picture(io.BytesIO(final_image_buf.getbuffer()), width=Inches(5))
                    else:
                        doc.add_picture(final_image_buf, width=Inches(5))

        # Save DOCX
        output_docx = "Generated_Research_Paper.docx"
        doc.save(output_docx)

        # Try docx -> pdf with docx2pdf if available (Windows + Word)
        output_pdf = "Generated_Research_Paper.pdf"
        converted = False
        try:
            from docx2pdf import convert
            convert(output_docx, output_pdf)
            converted = True
        except Exception:
            converted = False

        # fallback PDF: text-only
        if not converted:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(output_pdf, pagesize=letter)
            textobj = c.beginText(40, 750)
            textobj.setFont("Helvetica", 10)
            for h, cont in paper_sections.items():
                textobj.textLine(h)
                for ln in cont.split("\n"):
                    textobj.textLine(ln[:200])
                textobj.textLine(" ")
            c.drawText(textobj)
            c.save()

        # provide downloads
        with open(output_docx, "rb") as f:
            st.download_button("Download DOCX", f, file_name=output_docx)
        with open(output_pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name=output_pdf)

        st.success("Done. Inspect the DOCX for styled output. If headings not detected, try using simpler separate-line headings in your template or paste your headings in Format step.")
