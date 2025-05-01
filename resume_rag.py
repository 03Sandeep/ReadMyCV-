import gradio as gr
import ollama
from PyPDF2 import PdfReader
import io
import re

# 1. Pre-load model for faster responses
ollama.generate(model='phi3', prompt="")  # Warm-up

# 2. PDF Processing
def extract_text(file_obj):
    """Extract text from first 3 pages of PDF"""
    try:
        pdf_reader = PdfReader(io.BytesIO(file_obj))
        return "\n".join([page.extract_text() or "" for page in pdf_reader.pages[:3]])
    except Exception as e:
        return f"PDF Error: {str(e)}"

# 3. Core AI Functions
def build_prompt(resume_text, question):
    """Dynamic prompt builder based on question type"""
    resume_chunk = resume_text[:2000]  # Optimize for CPU
    
    # Special handling for different question types
    if re.search(r'\b(name|who is)\b', question.lower()):
        return f"""Extract ONLY the full name from:
        {resume_chunk}
        RULES: Return "First Last" format or "Not found"
        NAME:"""
    elif re.search(r'\b(skill|ability|expertise)\b', question.lower()):
        return f"""List ALL technical/hard skills from:
        {resume_chunk}
        RULES: Comma-separated, no explanations
        SKILLS:"""
    else:  # General case
        return f"""Answer concisely using ONLY this resume:
        {resume_chunk}
        QUESTION: {question}
        RULES: If unsure, say "Not specified"
        ANSWER:"""

def validate_response(response, question):
    """Post-process AI output"""
    response = response.strip()
    
    # Fix common errors
    if not response or "I don't know" in response:
        return "Not specified in resume"
    
    # Validate names
    if re.search(r'\b(name|who is)\b', question.lower()):
        if not re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+$', response):
            return "Name not detected"
    
    # Validate numbers for experience questions
    if re.search(r'\b(year|experience|age)\b', question.lower()):
        if not any(c.isdigit() for c in response):
            return "Duration not specified"
    
    return response

# 4. Main Analysis Function
def analyze_resume(file_obj, question):
    if not file_obj:
        return "Upload a PDF first"
    
    resume_text = extract_text(file_obj)
    if "Error" in resume_text:
        return resume_text
        
    prompt = build_prompt(resume_text, question)
    
    response = ollama.generate(
        model='phi3',
        prompt=prompt,
        options={
            'temperature': 0.1,
            'num_ctx': 1024,
            'num_threads': 4  # Optimize for CPU cores
        },
        stream=False
    )
    
    return validate_response(response['response'], question)

# 5. Gradio Interface
with gr.Blocks(title="Resume Analyzer") as demo:
    gr.Markdown("## 🔍 Ask Anything About The Resume")
    with gr.Row():
        pdf = gr.File(label="Upload Resume (3 pages max)", 
                    file_types=[".pdf"],
                    type="binary")
        question = gr.Textbox(label="Your Question",
                            placeholder="e.g. 'What are their top skills?'")
    submit = gr.Button("Analyze", variant="primary")
    output = gr.Textbox(label="Answer", interactive=False)
    
    submit.click(
        fn=analyze_resume,
        inputs=[pdf, question],
        outputs=output,
        api_name="analyze"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)