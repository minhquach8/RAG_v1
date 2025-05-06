import os
import re
import time
import logging
from dotenv import load_dotenv

# Set up logging configuration
def setup_logging(log_file, log_level=logging.INFO):
    """
    Configure logging to write to both a file and the console.
    
    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level (e.g., logging.INFO).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Clear any existing handlers to avoid duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger

# Manage log file by checking its age and refreshing if needed
def manage_log_file(log_file, max_age, logger):
    """
    Check the age of the log file and refresh it if older than max_age (in seconds).
    
    Args:
        log_file (str): Path to the log file.
        max_age (int): Maximum age of the log file in seconds.
        logger (logging.Logger): Logger instance to reconfigure after refresh.
    """
    if os.path.exists(log_file):
        file_age = time.time() - os.path.getmtime(log_file)
        if file_age > max_age:
            logger.info(f"Log file {log_file} is older than {max_age} seconds. Creating new log file...")
            os.remove(log_file)
            with open(log_file, "a") as f:
                f.write(f"New log file created at {time.ctime()}\n")
            # Reconfigure logging after file refresh
            setup_logging(log_file)
            logger.info("Logging reconfigured after file refresh.")

# Clean LaTeX symbols from text
def clean_latex(text):
    """
    Remove LaTeX formatting (e.g., $...$) from the text.
    
    Args:
        text (str): Input text containing LaTeX symbols.
    
    Returns:
        str: Cleaned text with LaTeX symbols removed.
    """
    return re.sub(r"\$\\mathrm\{(.*?)\}\$", r"\1", text)

# Clean extra spaces and newlines from text
def clean_text(text):
    """
    Replace newlines with spaces and normalize multiple spaces into a single space.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Remove references and metadata from text
def remove_references(text):
    """
    Remove references, citations, and metadata (e.g., copyright notices) from the text.
    
    Args:
        text (str): Input text containing references.
    
    Returns:
        str: Text with references and metadata removed.
    """
    # Remove content after "References" section
    text = re.split(r"(?i)\bReferences\b", text)[0]
    # Remove citation patterns like "[number] Author, et al."
    text = re.sub(r"\[\d+\]\s*[A-Za-z\s,]+et al\..*?(?=\[\d+\]|\Z)", "", text)
    # Remove patterns like "Author, et al. Title [J]."
    text = re.sub(r"[A-Za-z\s,]+et al\.\s*.*?(\[J\]\.\s*|\d{4})", "", text)
    # Remove metadata and copyright notices
    text = re.sub(r"Permission to make digital or hard copies.*?(?=\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"Copyrights for components.*?(?=\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"ACM Reference Format:.*?(?=\n\d|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"Request permissions from permissions@acm\.org.*?(?=\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"© \d{4} Copyright held by the owner/author\(s\)\..*?(?=\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"Publication rights licensed to ACM.*?(?=\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"ACM ISBN \d+-\d+-\d+-\d+-\d+/\d+/\d+.*?(?=\n|\Z)", "", text, flags=re.DOTALL)
    # Remove citation numbers like [number]
    text = re.sub(r"\[\d+\](?:,\s*\[\d+\])*", "", text)
    return text.strip()

# Save processing progress to a file
def save_progress(last_processed_file, progress_file="progress.txt"):
    """
    Save the name of the last processed file to track progress.
    
    Args:
        last_processed_file (str): Name of the last processed file.
        progress_file (str): Path to the progress file.
    """
    logger = logging.getLogger(__name__)
    with open(progress_file, "w") as f:
        f.write(last_processed_file)
    logger.info(f"Progress saved: Last processed file is {last_processed_file}")

# Load processing progress from a file
def load_progress(progress_file="progress.txt"):
    """
    Load the name of the last processed file to resume processing.
    
    Args:
        progress_file (str): Path to the progress file.
    
    Returns:
        str: Name of the last processed file, or None if not found.
    """
    logger = logging.getLogger(__name__)
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            last_processed_file = f.read().strip()
        logger.info(f"Loaded progress: Last processed file was {last_processed_file}")
        return last_processed_file
    return None

# Load Hugging Face token from .env file
def load_hf_token():
    """
    Load the Hugging Face token from the .env file.
    
    Returns:
        str: Hugging Face token.
    
    Raises:
        ValueError: If the token is not found in the .env file.
    """
    logger = logging.getLogger(__name__)
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Hugging Face token (HF_TOKEN) not found.")
        raise ValueError("Hugging Face token (HF_TOKEN) not found. Please set it in .env file: HF_TOKEN='your_token'")
    return hf_token