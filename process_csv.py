import pandas as pd
import logging
import os
import torch
import time
from datetime import datetime
import sys
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv

# Configure logging with rotation every 30 minutes
def setup_logging(log_file):
    """
    Configure logging to write to both a file and the console.
    
    Args:
        log_file (str): Path to the log file.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Clear any existing handlers to avoid duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger

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

def load_hf_token(logger):
    """
    Load the Hugging Face token from the .env file.
    
    Args:
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        str: Hugging Face token.
    
    Raises:
        ValueError: If the token is not found in the .env file.
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Hugging Face token (HF_TOKEN) not found.")
        raise ValueError("Hugging Face token (HF_TOKEN) not found. Please set it in .env file: HF_TOKEN='your_token'")
    return hf_token

def initialize_llm(model_name, hf_token, device, max_new_tokens, logger):
    """
    Initialize the language model (LLM) for extracting disease names.
    
    Args:
        model_name (str): Name of the language model.
        hf_token (str): Hugging Face token for authentication.
        device (str): Device to run the model on (e.g., 'mps', 'cpu').
        max_new_tokens (int): Maximum number of tokens to generate.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        HuggingFacePipeline: Initialized LLM pipeline.
    """
    logger.info("Initializing LLM for disease extraction")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            device=-1 if device == "cpu" else None,
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def extract_disease_name(text, llm, logger):
    """
    Use the LLM to extract the disease name from a lengthy description, with a fallback for known viral diseases.
    
    Args:
        text (str): The full description in the Disease column.
        llm (HuggingFacePipeline): The language model for extraction.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        str: The simplified disease name, or the original text if extraction fails.
    """
    if pd.isna(text) or "Error" in text:
        return text
    
    # Updated prompt to explicitly include viral infections as diseases
    prompt = (
        "You are an expert in medical research. Given the following description, extract the name of the disease in a concise form (e.g., 'Parkinson's disease', 'clinical depression', 'COVID-19'). "
        "If the description mentions a virus or pandemic (e.g., 'COVID-19 pandemic', 'HIV virus'), treat the viral infection as the disease (e.g., return 'COVID-19' for 'COVID-19 pandemic'). "
        "Return only the disease name, nothing else. If no disease is identified, return 'unknown disease'.\n\n"
        f"Description: {text}\n\n"
        "Disease name:"
    )
    
    try:
        response = llm.invoke(prompt)
        # Extract the disease name from the response (remove the prompt part)
        disease_name = response.split("Disease name:")[-1].strip().lower()
        if not disease_name:
            disease_name = "unknown disease"
    except Exception as e:
        logger.error(f"Error extracting disease name from text: {str(e)}")
        disease_name = "unknown disease"
    
    # Fallback: Check for known viral diseases if LLM returns "unknown disease"
    if disease_name == "unknown disease":
        known_viral_diseases = [
            ("covid-19", ["covid-19", "coronavirus", "sars-cov-2"]),
            ("hiv/aids", ["hiv", "aids"]),
            ("ebola", ["ebola"]),
            ("influenza", ["influenza", "flu"]),
            ("zika", ["zika"]),
            ("dengue", ["dengue"]),
            ("hepatitis", ["hepatitis"]),
        ]
        text_lower = text.lower()
        for disease, keywords in known_viral_diseases:
            if any(keyword in text_lower for keyword in keywords):
                disease_name = disease
                logger.info(f"Fallback: Identified {disease} from keywords in description")
                break
    
    logger.info(f"Extracted disease name: {disease_name}")
    return disease_name

def read_csv_file(file_path, logger):
    """Read CSV file and handle potential errors."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        logger.info(f"Successfully read CSV file: {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading CSV file: {e}")
        raise

def extract_error_papers(df, output_file, logger):
    """Extract papers with 'Error' in any column and save to a text file."""
    try:
        error_papers = df[df.apply(lambda row: row.str.contains('Error', na=False).any(), axis=1)]['Paper']
        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in error_papers:
                f.write(f"{paper}\n")
        logger.info(f"Saved {len(error_papers)} papers with errors to {output_file}")
        return error_papers
    except Exception as e:
        logger.error(f"Error extracting papers with errors: {e}")
        raise

def analyze_diseases_and_models(df, llm, logger):
    """Analyze diseases and AI models, extracting unique names."""
    try:
        # Preprocess the Disease column to extract simplified disease names using the LLM
        df['Disease'] = df['Disease'].apply(lambda x: extract_disease_name(x, llm, logger))
        logger.info("Extracted simplified disease names from Disease column")
        
        # Extract unique disease names
        diseases = df['Disease'].dropna().str.strip().unique().tolist()
        
        # Extract unique AI model names
        ai_models = df['AI_Model'].dropna().str.split(', ').explode().str.strip().unique().tolist()
        
        logger.info("Extracted unique disease and AI model names")
        return diseases, ai_models
    except Exception as e:
        logger.error(f"Error analyzing diseases and AI models: {e}")
        raise

def save_names_to_csv(disease_names, model_names, disease_output_file, model_output_file, logger):
    """
    Save disease and AI model names to separate CSV files.
    
    Args:
        disease_names (list): List of unique disease names.
        model_names (list): List of unique AI model names.
        disease_output_file (str): Path to the CSV file for disease names.
        model_output_file (str): Path to the CSV file for model names.
        logger (logging.Logger): Logger instance for logging messages.
    """
    try:
        # Save disease names
        df_diseases = pd.DataFrame({'Name': disease_names})
        df_diseases.to_csv(disease_output_file, index=False, encoding='utf-8')
        logger.info(f"Saved disease names to {disease_output_file}")

        # Save model names
        df_models = pd.DataFrame({'Name': model_names})
        df_models.to_csv(model_output_file, index=False, encoding='utf-8')
        logger.info(f"Saved model names to {model_output_file}")
    except Exception as e:
        logger.error(f"Error saving names to CSV: {e}")
        raise

def print_analysis(disease_names, model_names, logger):
    """Print analysis results in the requested format."""
    try:
        logger.info("\nDisease Names:")
        print("\nDisease Names:")
        for disease in disease_names:
            message = f"Bệnh: {disease}"
            logger.info(message)
            print(message)
        
        logger.info("\nAI Model Names:")
        print("\nAI Model Names:")
        for model in model_names:
            message = f"Model: {model}"
            logger.info(message)
            print(message)
    except Exception as e:
        logger.error(f"Error printing analysis results: {e}")
        raise

def main():
    """Main function to process CSV file and perform analysis."""
    # Setup logging
    LOG_FILE = "csv_processor.log"
    LOG_MAX_AGE = 1800  # 30 minutes in seconds
    logger = setup_logging(LOG_FILE)
    
    # Refresh log file if needed
    manage_log_file(LOG_FILE, LOG_MAX_AGE, logger)
    
    logger.info("Starting CSV processing program")
    
    try:
        # File paths
        csv_file = "article_answers.csv"
        error_output_file = "error_papers.txt"
        disease_output_file = "disease_names.csv"
        model_output_file = "model_names.csv"
        
        # LLM setup (using same settings as PDF pipeline)
        MODEL_NAME = "Qwen/Qwen3-8B"
        MAX_NEW_TOKENS = 50  # Smaller value since we only need a short disease name
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load Hugging Face token
        hf_token = load_hf_token(logger)
        
        # Initialize LLM
        llm = initialize_llm(MODEL_NAME, hf_token, device, MAX_NEW_TOKENS, logger)
        
        # Read CSV
        df = read_csv_file(csv_file, logger)
        
        # Extract papers with errors
        extract_error_papers(df, error_output_file, logger)
        
        # Analyze diseases and AI models
        disease_names, model_names = analyze_diseases_and_models(df, llm, logger)
        
        # Save names to separate CSV files
        save_names_to_csv(disease_names, model_names, disease_output_file, model_output_file, logger)
        
        # Print results
        print_analysis(disease_names, model_names, logger)
        
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error(f"Program failed: {e}")
        raise

if __name__ == "__main__":
    main()