import os
import io
import json
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime
import urllib.parse

# Load API Keys
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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").strip()
POLLINATIONS_KEY = "nL9VR9GzFRabl2Id"

def generate_text_grok(prompt: str, max_tokens: int = 2000) -> str:
    """Generate text using OpenRouter's Grok model"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Research Assistant"
    }
    
    body = {
        "model": "x-ai/grok-4-fast:free",  # Using the correct model name
        "messages": [
            {
                "role": "system",
                "content": "You are an expert academic researcher and writer. Generate high-quality, detailed research content that is specific, informative, and well-structured. Always write comprehensive content."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        if 'choices' not in data or not data['choices']:
            return "[Error: No response from AI model]"
        
        content = data['choices'][0]['message']['content'].strip()
        return content if content else "[Error: Empty response]"
        
    except requests.exceptions.RequestException as e:
        return f"[API Error: {str(e)}]"
    except Exception as e:
        return f"[Generation Error: {str(e)}]"

def generate_image_pollinations(prompt: str, width: int = 1024, height: int = 768) -> bytes:
    """Generate images using Pollinations AI"""
    try:
        # Clean prompt
        clean_prompt = re.sub(r'[^\w\s\-.,]', '', prompt)
        enhanced_prompt = f"professional academic research illustration: {clean_prompt}, clean design, high quality, detailed"
        
        # Use direct URL approach that works
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&model=flux&enhance=true"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        if response.headers.get('content-type', '').startswith('image/'):
            return response.content
        else:
            raise RuntimeError("Invalid response format")
            
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

def create_professional_placeholder(text: str, width=800, height=450) -> io.BytesIO:
    """Create professional placeholder images"""
    img = Image.new("RGB", (width, height), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
    except:
        font_title = font_text = ImageFont.load_default()
    
    # Border
    draw.rectangle([10, 10, width-10, height-10], outline=(70, 130, 180), width=3)
    
    # Title
    title = "Research Illustration"
    draw.text((50, 30), title, font=font_title, fill=(25, 25, 112))
    
    # Content
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if len(test_line) <= 60:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    y_offset = 80
    for line in lines[:8]:
        draw.text((30, y_offset), line, font=font_text, fill=(60, 60, 60))
        y_offset += 30
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

def generate_enhanced_graph(description: str, graph_type: str = "auto") -> io.BytesIO:
    """Generate professional graphs"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine graph type
    if graph_type == "auto":
        desc_lower = description.lower()
        if "bar" in desc_lower or "comparison" in desc_lower:
            graph_type = "bar"
        elif "line" in desc_lower or "trend" in desc_lower or "time" in desc_lower:
            graph_type = "line"
        elif "pie" in desc_lower or "distribution" in desc_lower:
            graph_type = "pie"
        else:
            graph_type = "bar"
    
    # Sample data
    if "accuracy" in description.lower() or "performance" in description.lower():
        categories = ["Baseline", "Method A", "Method B", "Proposed"]
        values = [0.72, 0.81, 0.85, 0.92]
        ylabel = "Accuracy"
    elif "time" in description.lower() or "speed" in description.lower():
        categories = ["Original", "Optimized", "Enhanced", "Final"]
        values = [120, 85, 60, 35]
        ylabel = "Processing Time (ms)"
    else:
        categories = ["Category A", "Category B", "Category C", "Category D"]
        values = [25, 42, 38, 55]
        ylabel = "Value"
    
    # Create graph
    if graph_type == "bar":
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Categories")
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                   f'{value:.2f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom')
    
    elif graph_type == "line":
        x_vals = list(range(1, len(values) + 1))
        ax.plot(x_vals, values, marker='o', linewidth=3, markersize=8, color='#3498db')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time/Steps")
        ax.grid(True, alpha=0.3)
    
    elif graph_type == "pie":
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        ax.pie(values, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax.set_title(f"Research Data: {description[:40]}{'...' if len(description) > 40 else ''}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", dpi=300, bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer

def create_research_document(topic: str, sections: dict) -> Document:
    """Create professional research document"""
    doc = Document()
    
    # Title
    title_para = doc.add_heading(topic, level=0)
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Author placeholder
    author_para = doc.add_paragraph("Author Name")
    author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_para.runs[0].italic = True
    
    # Institution
    inst_para = doc.add_paragraph("Institution Name")
    inst_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    inst_para.runs[0].italic = True
    
    # Date
    date_para = doc.add_paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Spacing
    doc.add_paragraph()
    
    # Add sections
    for heading, content in sections.items():
        if not content.startswith("[Error") and content.strip():
            doc.add_heading(heading, level=1)
            
            # Split content into paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            for para_text in paragraphs:
                if para_text:
                    para = doc.add_paragraph(para_text)
                    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    
    return doc

def generate_research_content(topic: str, headings: list) -> dict:
    """Generate comprehensive research content"""
    
    enhanced_prompts = {
        "Abstract": f"""Write a comprehensive academic abstract for a research paper titled: "{topic}". 
        The abstract should be 200-250 words and include:
        - Background and context of {topic}
        - Research objectives and questions
        - Methodology approach
        - Key findings and results
        - Implications and conclusions
        Write it as a complete, professional abstract that could be published.""",
        
        "Introduction": f"""Write a detailed introduction section for a research paper on "{topic}".
        Include:
        - Background context and importance of {topic}
        - Current state of the field
        - Existing challenges and limitations
        - Research gap being addressed
        - Clear research objectives
        - Paper structure overview
        Write 400-500 words with specific details about {topic}.""",
        
        "Literature Review": f"""Write a comprehensive literature review for "{topic}".
        Cover:
        - Historical development of {topic}
        - Current research approaches and methodologies
        - Key studies and their contributions
        - Comparative analysis of different approaches
        - Identified gaps and limitations
        - How this research addresses these gaps
        Include realistic citations like [1], [2], [3]. Write 350-400 words.""",
        
        "Methodology": f"""Write a detailed methodology section for "{topic}" research.
        Describe:
        - Overall research approach and design
        - Data collection methods and sources
        - Tools, technologies, and frameworks used
        - Implementation details and procedures
        - Evaluation metrics and criteria
        - Experimental setup or system architecture
        Make it specific to {topic}. Write 300-350 words.""",
        
        "Results": f"""Write a comprehensive results section for "{topic}" research.
        Present:
        - Key experimental outcomes and findings
        - Performance metrics and measurements
        - Comparative analysis with existing methods
        - Statistical analysis and significance
        - Visual data representation (reference to figures/tables)
        - Interpretation of results
        Write 300-350 words with specific results for {topic}.""",
        
        "Discussion": f"""Write an analytical discussion section for "{topic}" research.
        Analyze:
        - Interpretation of findings and their significance
        - Comparison with related work and literature
        - Practical implications for the {topic} field
        - Limitations and potential biases
        - Threats to validity
        - Future research opportunities
        Write 250-300 words.""",
        
        "Conclusion": f"""Write a strong conclusion for "{topic}" research.
        Summarize:
        - Main contributions and achievements
        - Key findings and their importance
        - Practical impact on the {topic} field
        - Limitations acknowledged
        - Future research directions
        - Broader implications and applications
        Write 200-250 words.""",
        
        "References": f"""Generate 10-12 realistic academic references for "{topic}".
        Include a mix of:
        - Recent journal papers (2020-2024)
        - Conference proceedings
        - Technical reports
        - Book chapters
        Format them properly as: Author, A.A. (Year). Title. Journal/Conference, Volume(Issue), pages.
        Make them relevant and realistic for {topic}."""
    }
    
    paper_sections = {}
    
    for heading in headings:
        # Find best matching prompt
        prompt_key = heading
        for key in enhanced_prompts.keys():
            if key.lower() in heading.lower() or heading.lower() in key.lower():
                prompt_key = key
                break
        
        prompt = enhanced_prompts.get(prompt_key, 
            f"Write a comprehensive academic '{heading}' section for a research paper about '{topic}'. "
            f"Make it detailed, professional, and specific to {topic}. Write 300-400 words.")
        
        content = generate_text_grok(prompt, max_tokens=2000)
        
        # Validate and retry if needed
        if len(content) < 100 and not content.startswith("["):
            retry_prompt = f"Write detailed academic content about {topic} for the {heading} section. Be specific and comprehensive. " + prompt
            content = generate_text_grok(retry_prompt, max_tokens=2000)
        
        paper_sections[heading] = content.strip()
    
    return paper_sections

# Streamlit UI
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>AI Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>Generate comprehensive research papers with AI-powered content and visuals</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("System Status")
    st.success("Text Generation: Ready")
    st.success("Image Generation: Ready") 
    st.success("Document Creation: Ready")

# Main interface
topic = st.text_input(
    "Research Topic", 
    placeholder="Enter your research topic (e.g., Machine Learning in Healthcare)",
    help="Be specific for better results"
)

# Document structure
structure_option = st.selectbox(
    "Document Structure",
    ["Standard Academic Paper", "Custom Sections"]
)

if structure_option == "Standard Academic Paper":
    user_headings = ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion", "References"]
else:
    headings_input = st.text_area(
        "Custom section headings (one per line):",
        value="Abstract\nIntroduction\nLiterature Review\nMethodology\nResults\nDiscussion\nConclusion\nReferences",
        height=150
    )
    user_headings = [h.strip() for h in headings_input.split('\n') if h.strip()]

# Visual content options
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Research Figures")
    fig_option = st.selectbox("Figures", ["None", "Generate", "Upload"])
    fig_prompt = ""
    if fig_option == "Generate":
        fig_prompt = st.text_input("Describe figure:", placeholder="system architecture, process flow")

with col2:
    st.subheader("Data Graphs")
    graph_option = st.selectbox("Graphs", ["None", "Generate", "Upload"])
    graph_prompt = ""
    graph_type = "auto"
    if graph_option == "Generate":
        graph_prompt = st.text_input("Describe data:", placeholder="performance metrics, comparison")
        graph_type = st.selectbox("Type", ["auto", "bar", "line", "pie"])

with col3:
    st.subheader("Cover Image")
    cover_option = st.selectbox("Cover", ["None", "Generate", "Upload"])
    cover_prompt = ""
    if cover_option == "Generate":
        cover_prompt = st.text_input("Describe cover:", placeholder="research theme visualization")

# Generate button
if st.button("Generate Research Paper", type="primary"):
    if not topic.strip():
        st.error("Please enter a research topic!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Generate content
            status_text.text("Generating research content...")
            progress_bar.progress(20)
            
            paper_sections = generate_research_content(topic, user_headings)
            
            # Check if generation was successful
            successful_sections = sum(1 for content in paper_sections.values() 
                                    if not content.startswith("[") and len(content) > 50)
            
            if successful_sections == 0:
                st.error("Failed to generate content. Please try again.")
                st.stop()
            
            progress_bar.progress(50)
            
            # Generate visuals
            status_text.text("Creating visual content...")
            generated_images = []
            
            # Generate figure
            if fig_option == "Generate" and fig_prompt:
                try:
                    img_bytes = generate_image_pollinations(fig_prompt)
                    generated_images.append(("Research Figure", io.BytesIO(img_bytes)))
                    st.success("Figure generated!")
                except Exception as e:
                    st.warning(f"Using placeholder for figure: {str(e)}")
                    generated_images.append(("Research Figure", create_professional_placeholder(fig_prompt)))
            
            # Generate cover
            if cover_option == "Generate" and cover_prompt:
                try:
                    cover_bytes = generate_image_pollinations(cover_prompt + ", academic research style")
                    generated_images.append(("Cover Image", io.BytesIO(cover_bytes)))
                    st.success("Cover generated!")
                except Exception as e:
                    st.warning(f"Using placeholder for cover: {str(e)}")
                    generated_images.append(("Cover Image", create_professional_placeholder(cover_prompt)))
            
            progress_bar.progress(70)
            
            # Generate graph
            if graph_option == "Generate" and graph_prompt:
                try:
                    graph_buffer = generate_enhanced_graph(graph_prompt, graph_type)
                    generated_images.append(("Data Graph", graph_buffer))
                    st.success("Graph generated!")
                except Exception as e:
                    st.warning(f"Graph generation failed: {str(e)}")
            
            progress_bar.progress(85)
            
            # Create document
            status_text.text("Creating document...")
            
            doc = create_research_document(topic, paper_sections)
            
            # Add images
            if generated_images:
                doc.add_heading("Figures and Visualizations", level=1)
                
                for img_name, img_buffer in generated_images:
                    try:
                        img_buffer.seek(0)
                        doc.add_picture(img_buffer, width=Inches(6))
                        
                        caption = doc.add_paragraph(f"Figure: {img_name}")
                        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        caption.runs[0].italic = True
                        
                        doc.add_paragraph()
                        
                    except Exception as e:
                        st.warning(f"Could not insert {img_name}")
            
            progress_bar.progress(95)
            
            # Save document
            status_text.text("Finalizing...")
            
            safe_topic = re.sub(r'[^\w\s-]', '', topic).strip()[:25]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"Research_Paper_{safe_topic.replace(' ', '_')}_{timestamp}.docx"
            
            doc.save(filename)
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            # Download
            with open(filename, "rb") as file:
                st.download_button(
                    label="Download Research Paper",
                    data=file,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            st.success(f"Generated research paper with {successful_sections} sections!")
            
            # Preview
            with st.expander("Content Preview"):
                for heading, content in paper_sections.items():
                    if not content.startswith("[") and len(content) > 50:
                        st.subheader(heading)
                        preview = content[:400] + "..." if len(content) > 400 else content
                        st.write(preview)
                        st.markdown("---")
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")

st.markdown("---")
st.markdown("*AI Research Assistant - Generating quality research papers with AI*")