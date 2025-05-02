# Configuration settings for the PDF processing and RAG pipeline
import os

# Directory path for PDF files
PDF_DIR = os.path.expanduser("~/Documents/PhD/CODING/Collect_All_PDFs_ENDNOTE/ALL_PDFs")

# Output file for storing results
OUTPUT_FILE = os.path.join(os.getcwd(), "article_answers.csv")

# Log file settings
LOG_FILE = "processing_log.txt"
LOG_MAX_AGE = 1800  # Maximum age of log file in seconds (30 minutes)

# Model and pipeline settings
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 300
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
RETRIEVER_K = 15
RETRIEVER_THRESHOLD = 0.5

# List of questions to extract from each paper
QUESTIONS_LIST = [
    {
        "question": "Who are the authors, and where and when was the article published?",
        "is_short_answer": False,
        "feature": "Authors_Publication",
    },
    {
        "question": "What is the DOI of the article?",
        "is_short_answer": False,
        "feature": "DOI",
    },
    {
        "question": "How was the data collected for this research, including sources, methods, and scale if available?",
        "is_short_answer": False,
        "feature": "Data_Collection",
    },
    {
        "question": "Is the data used in this research publicly accessible? If not, what are the restrictions or privacy measures applied?",
        "is_short_answer": False,
        "feature": "Data_Availability",
    },
    {
        "question": "What AI model is used in this research?",
        "is_short_answer": False,
        "feature": "AI_Model",
    },
    {
        "question": "What are the main findings of the research?",
        "is_short_answer": False,
        "feature": "Main_Findings",
    },
    {
        "question": "What are the advantages and disadvantages of the research?",
        "is_short_answer": False,
        "feature": "Pros_Cons",
    },
    {
        "question": "What methodology was used in this research to construct a large model, including details on data collection and specific steps if possible?",
        "is_short_answer": False,
        "feature": "Methodology",
    },
    {
        "question": "What are the future directions proposed in the research?",
        "is_short_answer": False,
        "feature": "Future_Directions",
    },
    {
        "question": "What disease or condition is the research focused on?",
        "is_short_answer": False,
        "feature": "Disease",
    },
]
