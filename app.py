import os
import io
import time
import base64
import json
import re
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

# ---- load keys ----
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
STABLE_HORDE_KEY = os.getenv("STABLE_HORDE_KEY", "0000000000").strip()
# Note: Gemini/Groq commented out (user request). If you want to add them, put keys in .env and uncomment.
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---- helpers: Mistral text generation ----
def generate_text_mistral(prompt: str, model: str = "mistral-large-latest", max_tokens: int = 1200, temperature: float = 0.2) -> str:
    """ Call Mistral chat completions endpoint. Uses the chat format from Mistral docs.
    Docs / quickstart: https://docs.mistral.ai/quickstart (see API examples).
    :contentReference[oaicite:3]{index=3}
    """
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is not set in .env")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Mistral generation failed: {e} - {resp.text[:400]}")
    j = resp.json()
    # Expect: choices[0].message.content
    try:
        return j["choices"][0]["message"]["content"].strip()
    except Exception:
        # fallback
        return json.dumps(j)[:2000]

# ---- helpers: Stable Horde image generation (with longer timeout + fallback) ----
def generate_image_horde(prompt: str, width: int = 512, height: int = 512, model: str = "stable_diffusion", timeout_s: int = 300) -> bytes:
    api_key = STABLE_HORDE_KEY or "0000000000"
    submit_url = "https://stablehorde.net/api/v2/generate/async"
    status_url_base = "https://stablehorde.net/api/v2/generate/status/"
    headers = {"apikey": api_key, "Client-Agent": "AIResearchAssistant/1.0", "Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "params": {"width": width, "height": height, "steps": 20, "n": 1, "cfg_scale": 7},
        "models": [model],
        "nsfw": False
    }
    try:
        r = requests.post(submit_url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError("Stable Horde didn't return job id")
    except Exception as e:
        raise RuntimeError(f"Stable Horde submit failed: {e}")

    # Polling up to timeout_s
    start = time.time()
    while time.time() - start < timeout_s:
        time.sleep(5)
        try:
            status_resp = requests.get(status_url_base + str(job_id), headers=headers, timeout=20)
            status_resp.raise_for_status()
            stt = status_resp.json()
            if stt.get("done") and stt.get("generations"):
                img_data = stt["generations"][0]["img"]
                if img_data.startswith("http"):
                    return requests.get(img_data).content
                return base64.b64decode(img_data)
        except Exception:
            pass
    # If timeout – fallback
    raise RuntimeError("Stable Horde image generation timed out")

# ---- small placeholder image/graph generators (fast local fallback) ----
def generate_placeholder_image(text: str, width=800, height=450) -> io.BytesIO:
    img = Image.new("RGB", (width, height), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        fnt = ImageFont.load_default()
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur + " " + w) > 45:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    y = 30
    for ln in lines[:10]:
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

# ---- docx helpers: insert/delete paragraphs safely ----
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

# sanitize model outputs: remove weird characters, reduce consecutive special chars
def sanitize_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[\r\t]+", " ", s)
    s = re.sub(r"\u200b", "", s)  # zero width
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", "", s)
    # collapse more than 3 non-word punctuation to single period
    s = re.sub(r"[^\w\s]{3,}", ".", s)
    # trim
    return s.strip()

# ---- UI ----
st.set_page_config(page_title="AI Research Assistant (Mistral)", layout="wide")
st.title("AI Research Assistant — Mistral + Stable Horde")
st.markdown("Quick: upload a DOCX template for best fidelity. If you only have a PDF template, upload it but DOCX gives better design matching.")

# 1) Topic
topic = st.text_input("Research topic / title", value="AI tools and technologies in day-to-day life")

# 2) Format / headings
st.subheader("Format / Structure")
have_format = st.radio("Do you have a preferred format/structure (headings)?", ("No","Yes"))
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
st.subheader("Template")
have_template = st.radio("Do you have a DOCX template? (recommended)", ("Yes","No"))
uploaded_template = None
template_choice = None
if have_template == "Yes":
    uploaded_template = st.file_uploader("Upload DOCX template (will insert text under headings)", type=["docx"])
else:
    template_choice = st.selectbox("Pick a default template style", ["Simple Academic","IEEE","APA","MLA"])
st.info("A simple DOCX template will be generated if you don't upload one.")

# 4) Resources
st.subheader("Resources (optional)")
res_choice = st.radio("Resources for reference?", ("No","Upload","Autogenerate"))
uploaded_resources = []
resources_description = ""
if res_choice == "Upload":
    uploaded_resources = st.file_uploader("Upload PDF/DOCX/TXT files (multiple allowed)", type=["pdf","docx","txt"], accept_multiple_files=True)
elif res_choice == "Autogenerate":
    resources_description = st.text_area("Describe resources you'd like auto-generated (e.g., '2 recent survey papers, 3 datasets')")

# 5) Images
st.subheader("Figures / Images")
img_choice = st.radio("Images", ("None","Upload","Autogenerate"))
uploaded_images = []
img_description = ""
if img_choice == "Upload":
    uploaded_images = st.file_uploader("Upload images", type=["png","jpg","jpeg"], accept_multiple_files=True)
elif img_choice == "Autogenerate":
    img_description = st.text_input("Describe the image(s) to generate (short)")

# 6) Graphs
st.subheader("Graphs")
graph_choice = st.radio("Graphs", ("None","Upload","Autogenerate"))
uploaded_graphs = []
graph_description = ""
if graph_choice == "Upload":
    uploaded_graphs = st.file_uploader("Upload graph images", type=["png","jpg","jpeg"], accept_multiple_files=True)
elif graph_choice == "Autogenerate":
    graph_description = st.text_input("Describe the graph to generate (bar/line/pie + short notes)")

# 7) Final image
st.subheader("Final output image (optional)")
final_choice = st.radio("Final image", ("None","Upload","Autogenerate"))
uploaded_final_image = None
final_description = ""
if final_choice == "Upload":
    uploaded_final_image = st.file_uploader("Upload final image", type=["png","jpg","jpeg"])
elif final_choice == "Autogenerate":
    final_description = st.text_input("Describe final image to generate (short)")

# Generate button
if st.button("Generate research paper"):
    if not topic:
        st.error("Please enter a research topic.")
    else:
        st.info("Generating content... please wait (will update progress).")
        # Build prompts for headings (smart mapping)
        prompts = {}
        for h in user_headings:
            low = h.lower()
            if "abstract" in low:
                prompts[h] = f"Write a concise academic abstract for a paper on: {topic}."
            elif "introduc" in low:
                prompts[h] = f"Write an introduction for a research paper on: {topic}. Keep it formal and cite common themes."
            elif "literature" in low or "related" in low:
                prompts[h] = f"Write a short literature review summarizing common approaches and gaps for: {topic}."
            elif "method" in low:
                prompts[h] = f"Write a methodology section sketch for research on {topic}, including datasets, experiments, and evaluation metrics."
            elif "result" in low:
                prompts[h] = f"Describe plausible results/findings and how they'd be presented for research on {topic}."
            elif "discuss" in low:
                prompts[h] = f"Write a discussion section reflecting implications, limitations, and future work for {topic}."
            elif "conclusion" in low:
                prompts[h] = f"Write a clear conclusion for {topic} summarizing key contributions and next steps."
            elif "refer" in low:
                prompts[h] = f"Provide 5 sample references (title, authors, year, venue) relevant to {topic}."
            else:
                prompts[h] = f"Write the section titled '{h}' for a research paper about {topic}."

        # generate content per heading
        paper_sections = {}
        for h, p in prompts.items():
            st.write(f"Generating: {h} ...")
            try:
                out = generate_text_mistral(p, max_tokens=900)
                out = sanitize_text(out)
            except Exception as e:
                out = f"[Error generating {h}: {e}]"
            paper_sections[h] = out

        # Summarize uploaded resources if any
        resource_summaries = []
        if res_choice == "Upload" and uploaded_resources:
            st.write("Summarizing uploaded resources...")
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
                        summary = generate_text_mistral(f"Summarize this resource briefly:\n\n{short}", max_tokens=400)
                        summary = sanitize_text(summary)
                    except Exception as e:
                        summary = f"[Error summarizing: {e}]"
                else:
                    summary = "[No text extracted]"
                resource_summaries.append((name, summary))
        elif res_choice == "Autogenerate" and resources_description:
            try:
                resource_text = generate_text_mistral(f"Generate 2 short resource summaries relevant to {topic}: {resources_description}", max_tokens=400)
                resource_summaries.append(("AI-generated", sanitize_text(resource_text)))
            except Exception as e:
                resource_summaries.append(("AI-generated", f"[Error: {e}]"))

        # Prepare images/graphs buffers
        images_to_insert = []
        if img_choice == "Upload" and uploaded_images:
            for im in uploaded_images:
                images_to_insert.append(im)
        elif img_choice == "Autogenerate" and img_description:
            st.write("Generating image via Stable Horde...")
            try:
                img_bytes = generate_image_horde(img_description)
                images_to_insert.append(io.BytesIO(img_bytes))
            except Exception as e:
                st.warning(f"Stable Horde failed: {e}")
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
            try:
                final_bytes = generate_image_horde(final_description)
                final_image_buf = io.BytesIO(final_bytes)
            except Exception as e:
                st.warning(f"Stable Horde failed: {e}")
                final_image_buf = generate_placeholder_image(final_description)

        # Load template or create simple doc
        working_docx = "uploaded_template.docx"
        if uploaded_template:
            with open(working_docx, "wb") as tmpf:
                tmpf.write(uploaded_template.getbuffer())
            doc = Document(working_docx)
        else:
            # create a nice simple DOCX template
            doc = Document()
            try:
                doc.add_heading(topic, level=0)
            except Exception:
                p = doc.add_paragraph()
                r = p.add_run(topic)
                r.bold = True
            doc.add_paragraph(f"Generated using Mistral (model) — topic: {topic}")

        # Try to find headings in template (paragraph.style.name startswith 'Heading' OR exact text match)
        paras = doc.paragraphs
        heading_indexes = {}
        for idx, p in enumerate(paras):
            txt = (p.text or "").strip()
            style_name = ""
            try:
                style_name = getattr(p.style, "name", "") or ""
            except Exception:
                style_name = ""
            if txt:
                # exact match with any of user_headings (case-insensitive)
                for h in user_headings:
                    if txt.lower() == h.lower() or txt.lower().startswith(h.lower()) or h.lower() in txt.lower():
                        heading_indexes[h] = idx
                        break
                # also check style name
                if not any(k==txt for k in heading_indexes.keys()) and style_name.lower().startswith("heading"):
                    # match the nearest heading name by similarity if any
                    for h in user_headings:
                        if h.lower() in txt.lower() or txt.lower() in h.lower():
                            heading_indexes[h] = idx
                            break

        # For each found heading, delete following non-heading paragraphs and insert generated
        # Note: operating on doc.paragraphs while deleting can be tricky; we re-fetch doc.paragraphs after deletes.
        for heading, idx in sorted(heading_indexes.items(), key=lambda x: x[1]):
            try:
                paras = doc.paragraphs
                anchor = paras[idx]
            except Exception:
                continue
            # remove until next heading occurrence
            j = idx + 1
            while True:
                paras = doc.paragraphs
                if j >= len(paras):
                    break
                txt = (paras[j].text or "").strip().lower()
                found_next = False
                for h2 in user_headings:
                    if txt == h2.lower() or txt.startswith(h2.lower()) or h2.lower() in txt:
                        found_next = True
                        break
                if found_next:
                    break
                # delete paragraph j
                try:
                    delete_paragraph(paras[j])
                except Exception:
                    pass
                # don't increment j because we removed paras[j]
            # insert generated content after anchor
            content = paper_sections.get(heading, "")
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
            if not blocks and content.strip():
                blocks = [content.strip()]
            insertion = anchor
            for block in blocks:
                insertion = insert_paragraph_after(insertion, text=block, style=anchor.style)

        # Append any sections not present already
        existing_text = " ".join([p.text for p in doc.paragraphs]).lower()
        for h, content in paper_sections.items():
            if h.lower() not in existing_text:
                # add heading
                try:
                    doc.add_heading(h, level=1)
                except Exception:
                    p = doc.add_paragraph()
                    r = p.add_run(h)
                    r.bold = True
                for block in [b for b in content.split("\n\n") if b.strip()]:
                    doc.add_paragraph(block)

        # Insert resource summaries near References if found, else at end
        refs_idx = None
        for idx, p in enumerate(doc.paragraphs):
            if "reference" in (p.text or "").lower():
                refs_idx = idx
                break
        if resource_summaries:
            if refs_idx is not None:
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

        # Insert Figures near Results heading if present
        results_idx = None
        for idx, p in enumerate(doc.paragraphs):
            if "result" in (p.text or "").lower():
                results_idx = idx
                break

        def _add_picture_to_doc(doc, img_buf, caption=None):
            try:
                doc.add_picture(img_buf, width=Inches(5))
                if caption:
                    doc.add_paragraph(caption)
            except Exception:
                # fallback: save temp and add
                tmpn = "tmp_img_for_doc.png"
                with open(tmpn, "wb") as wf:
                    wf.write(img_buf.getbuffer() if hasattr(img_buf, "getbuffer") else img_buf.read())
                doc.add_picture(tmpn, width=Inches(5))
                if caption:
                    doc.add_paragraph(caption)

        if images_to_insert or graphs_to_insert or final_image_buf:
            if results_idx is not None:
                anchor = doc.paragraphs[results_idx]
                ins = insert_paragraph_after(anchor, text="Figures", style=anchor.style)
                # images
                for im in images_to_insert:
                    if hasattr(im, "getbuffer"):
                        _add_picture_to_doc(doc, io.BytesIO(im.getbuffer()))
                    elif isinstance(im, io.BytesIO):
                        _add_picture_to_doc(doc, im)
                    else:  # file-like uploaded
                        _add_picture_to_doc(doc, io.BytesIO(im.read()))
                # graphs
                for gr in graphs_to_insert:
                    if hasattr(gr, "getbuffer"):
                        _add_picture_to_doc(doc, io.BytesIO(gr.getbuffer()))
                    else:
                        _add_picture_to_doc(doc, gr)
                # final
                if final_image_buf:
                    _add_picture_to_doc(doc, io.BytesIO(final_image_buf.getbuffer()) if hasattr(final_image_buf, "getbuffer") else final_image_buf)
            else:
                doc.add_heading("Figures", level=2)
                for im in images_to_insert:
                    if hasattr(im, "getbuffer"):
                        _add_picture_to_doc(doc, io.BytesIO(im.getbuffer()))
                    elif isinstance(im, io.BytesIO):
                        _add_picture_to_doc(doc, im)
                    else:
                        _add_picture_to_doc(doc, io.BytesIO(im.read()))
                for gr in graphs_to_insert:
                    if hasattr(gr, "getbuffer"):
                        _add_picture_to_doc(doc, io.BytesIO(gr.getbuffer()))
                    else:
                        _add_picture_to_doc(doc, gr)
                if final_image_buf:
                    _add_picture_to_doc(doc, io.BytesIO(final_image_buf.getbuffer()) if hasattr(final_image_buf, "getbuffer") else final_image_buf)

        # Save DOCX
        output_docx = "Generated_Research_Paper.docx"
        doc.save(output_docx)

        # Try docx -> pdf with docx2pdf (Windows + Word)
        output_pdf = "Generated_Research_Paper.pdf"
        converted = False
        try:
            from docx2pdf import convert
            convert(output_docx, output_pdf)
            converted = True
        except Exception:
            converted = False

        # fallback PDF: text-only summary (reportlab)
        if not converted:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            c = canvas.Canvas(output_pdf, pagesize=letter)
            textobj = c.beginText(40, 750)
            textobj.setFont("Helvetica", 10)
            for h, cont in paper_sections.items():
                textobj.textLine(h)
                for ln in cont.split("\n"):
                    # keep line length reasonable
                    for chunk in [ln[i:i+120] for i in range(0, len(ln), 120)]:
                        textobj.textLine(chunk)
                    textobj.textLine(" ")
            c.drawText(textobj)
            c.save()

        # downloads
        with open(output_docx, "rb") as f:
            st.download_button("Download DOCX", f, file_name=output_docx)
        with open(output_pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name=output_pdf)

        st.success("Done. Check DOCX for styled output. If headings are not detected, try using simpler separate-line headings in your template or paste the headings in the Format step.")