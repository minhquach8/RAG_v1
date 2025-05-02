import os
import fitz  # PyMuPDF
import time
import torch
import pandas as pd
import shutil  # Thêm để xóa thư mục
from transformers import logging as transformers_logging

# Import custom modules
import config
import utils
import pdf_processor
import rag_pipeline

# Suppress transformers logging warnings
transformers_logging.set_verbosity_error()

def main():
    """Main function to process PDFs and extract information using a RAG pipeline."""
    # Set up logging
    logger = utils.setup_logging(config.LOG_FILE)
    
    # Refresh log file if needed
    utils.manage_log_file(config.LOG_FILE, config.LOG_MAX_AGE)

    # Load Hugging Face token
    hf_token = utils.load_hf_token()

    # Determine device for model execution
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load list of PDF files
    pdf_files = [f for f in os.listdir(config.PDF_DIR) if f.endswith(".pdf")]
    pdf_files.sort()
    logger.info(f"Found {len(pdf_files)} PDF files in {config.PDF_DIR}")

    # Load processing progress
    last_processed_file = utils.load_progress()
    start_index = 0
    if last_processed_file:
        try:
            start_index = pdf_files.index(last_processed_file) + 1
            logger.info(f"Resuming from file {last_processed_file}, starting at index {start_index}")
        except ValueError:
            logger.warning(f"Last processed file {last_processed_file} not found in current list. Starting from the beginning.")
            start_index = 0

    # Initialize embeddings and LLM
    embeddings = rag_pipeline.initialize_embeddings("intfloat/e5-large-v2", device, logger)
    llm = rag_pipeline.initialize_llm(config.MODEL_NAME, hf_token, device, config.MAX_NEW_TOKENS, logger)

    # Process each PDF file
    processed_files = 0
    total_files = len(pdf_files) - start_index
    for i in range(start_index, len(pdf_files)):
        pdf_file = pdf_files[i]
        pdf_path = os.path.join(config.PDF_DIR, pdf_file)
        logger.info(f"Processing file {i + 1}/{len(pdf_files)}: {pdf_file}")
        processed_files += 1

        # Load PDF and extract metadata
        try:
            doc = fitz.open(pdf_path)
            max_pages = len(doc)
            logger.info(f"Total pages in PDF: {max_pages}")
            metadata = doc.metadata
            logger.info(f"Metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            continue

        # Extract DOI and authors
        doi = pdf_processor.extract_doi_from_pdf(pdf_path, logger)
        authors = pdf_processor.extract_authors(pdf_path, logger)

        # Load and preprocess PDF content
        try:
            documents = pdf_processor.load_and_preprocess_pdf(pdf_path, logger, utils.clean_latex, utils.clean_text)
        except Exception as e:
            logger.error(f"Error preprocessing PDF {pdf_path}: {str(e)}")
            continue

        # Create or load FAISS vector store
        pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
        faiss_index_path = os.path.join(os.getcwd(), f"faiss_index_{pdf_name}")  # Đường dẫn thư mục FAISS
        try:
            vectorstore = rag_pipeline.create_or_load_vectorstore(
                documents, embeddings, pdf_name, config.CHUNK_SIZE, config.CHUNK_OVERLAP, logger
            )
        except Exception as e:
            logger.error(f"Error creating/loading vectorstore for {pdf_path}: {str(e)}")
            continue

        # Create RAG pipeline
        try:
            qa_chain = rag_pipeline.create_rag_pipeline(
                llm, vectorstore, config.RETRIEVER_K, config.RETRIEVER_THRESHOLD, logger
            )
        except Exception as e:
            logger.error(f"Error creating RAG pipeline for {pdf_path}: {str(e)}")
            continue

        # Generate answers
        answers_dict = {"Paper": pdf_name}
        try:
            answers = rag_pipeline.generate_answers(
                qa_chain, vectorstore, config.QUESTIONS_LIST, doi, authors,
                config.RETRIEVER_K, config.RETRIEVER_THRESHOLD, utils.remove_references, logger
            )
            answers_dict.update(answers)
        except Exception as e:
            logger.error(f"Error generating answers for {pdf_path}: {str(e)}")
            continue

        # Save results to CSV
        logger.info(f"Appending results to {config.OUTPUT_FILE}")
        try:
            df = pd.DataFrame([answers_dict])
            required_columns = ["Paper"] + [q["feature"] for q in config.QUESTIONS_LIST]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "N/A"
            df = df[required_columns]
            df = df.replace(r"\n", " ", regex=True)
            if os.path.exists(config.OUTPUT_FILE):
                df.to_csv(config.OUTPUT_FILE, mode="a", header=False, index=False)
            else:
                df.to_csv(config.OUTPUT_FILE, mode="w", header=True, index=False)
            logger.info(f"Results appended successfully to {config.OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"Error appending results to {config.OUTPUT_FILE}: {str(e)}")

        # Save progress
        utils.save_progress(pdf_file)

        # Delete FAISS index folder after processing the PDF
        if os.path.exists(faiss_index_path):
            try:
                shutil.rmtree(faiss_index_path)
                logger.info(f"Deleted FAISS index folder: {faiss_index_path}")
            except Exception as e:
                logger.error(f"Error deleting FAISS index folder {faiss_index_path}: {str(e)}")

        # Pause after processing 50 files
        if processed_files % 50 == 0:
            logger.info(f"Processed {processed_files} files. Pausing for 5 minutes to cool down...")
            time.sleep(300)  # 5 minutes
            logger.info("Resuming processing...")

        # Ask to continue after 200 files
        if processed_files % 200 == 0:
            logger.info(f"Processed {processed_files} files out of {total_files} remaining.")
            while True:
                choice = input("Do you want to continue running? (y/n): ").strip().lower()
                if choice in ["y", "n"]:
                    break
                print("Please enter 'y' for yes or 'n' for no.")
            if choice == "n":
                logger.info("User chose to stop. Exiting...")
                break

    # Final cleanup: Delete any remaining FAISS index folders
    for folder in os.listdir(os.getcwd()):
        if folder.startswith("faiss_index_"):
            folder_path = os.path.join(os.getcwd(), folder)
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Final cleanup - Deleted FAISS index folder: {folder_path}")
            except Exception as e:
                logger.error(f"Final cleanup - Error deleting FAISS index folder {folder_path}: {str(e)}")

    logger.info("Processing completed.")

if __name__ == "__main__":
    main()