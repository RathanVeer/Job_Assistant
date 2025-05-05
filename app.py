import streamlit as st
from pypdf import PdfReader
import os
from langchain_groq import ChatGroq
#WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
model="llama3-70b-8192",
temperature=0,
groq_api_key = os.getenv("GROQ_API_KEY")
)

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text.strip()

def scrape_job_description(jd_link):
        
        loader = WebBaseLoader(jd_link)
        page_data = loader.load().pop().page_content


        prompt_extract = PromptTemplate.from_template(
                """
                ### SCARPED TEXT FROM WEBSITE
                {page_data}
                ### INSTRUCTION:
                The scrapped text is from the carrer's page of a website.
                Your job is to extract the job postings and return them in JSON format containing the
                following keys: 'role','experiencee','skills',and 'description',company_name.
                Only return the valid JSON (NO PREAMBLE).
                ### VALID JSON :
                """
        )

        chain_extract = prompt_extract | llm
        res = chain_extract.invoke(input={'page_data':page_data})
        return res.content

# Streamlit UI
st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")

st.title("📧 Cold Mail Generator")

# Sidebar for inputs
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    jd_link = st.text_input("Paste Job Description URL")

if uploaded_file and jd_link:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        jd_text = scrape_job_description(jd_link)

        prompt = f"""
        You are an expert assistant that writes cold emails to recruiters for job seekers.

        (**NO PREAMBLE**) . Output only the email body.

        Resume:
        ---
        {resume_text}
        ---

        Job Description:
        ---
        {jd_text}
        ---

        Write a short, personalized, and professional cold email to the recruiter. The email should:

        - Clearly express interest in the job role by linking with current experience.
        - Highlight 2–3 relevant strengths or experiences aligned with the JD.
        - Sound enthusiastic but professional.
        - Avoid copying the resume verbatim.
        - End with a polite call to action (e.g., request to connect or consider for referral).
        - Assume you do not know the recruiter's name.
        """
        output = llm.invoke(prompt).content
        lines = output.splitlines()
        updated_output = lines[1:]
        ans = "\n".join(updated_output)

        st.code(ans, language='markdown')

        st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px;
            text-align: center;
            font-size: 12px;
            color: #555;
            background-color: #f1f1f1;
        }
        </style>
        <div class="footer">
            Made with ❤️ by Rathan Veer
        </div>
    """, unsafe_allow_html=True)