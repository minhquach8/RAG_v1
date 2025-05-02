import fitz
import re
from langchain_community.document_loaders import PyMuPDFLoader

# Extract DOI from a PDF document
def extract_doi_from_pdf(pdf_path, logger):
    """
    Extract the DOI from a PDF by searching all pages for DOI patterns.
    
    Args:
        pdf_path (str): Path to the PDF file.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        str: Extracted DOI, or None if not found.
    """
    logger.info(f"Extracting DOI from {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            # Look for DOI patterns (e.g., "doi: 10.xxx" or "https://doi.org/10.xxx")
            doi_match = re.search(r"(?i)(?:doi\s*:\s*|https?://doi\.org/)(10\.\d{4,}(?:/[\w\-\.]+)+)", text)
            if doi_match:
                logger.info(f"DOI found: {doi_match.group(1)}")
                return doi_match.group(1)
            # Look for standalone DOI pattern (e.g., "10.xxx")
            doi_match = re.search(r"(?i)10\.\d{4,}(?:/[\w\-\.]+)+", text)
            if doi_match:
                logger.info(f"DOI found: {doi_match.group(0)}")
                return doi_match.group(0)
        logger.warning(f"No DOI found in {pdf_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting DOI from {pdf_path}: {str(e)}")
        return None

# Extract author names from a PDF document
def extract_authors(pdf_path, logger):
    """
    Extract author names from a PDF by searching the first page, all pages, and metadata.
    
    Args:
        pdf_path (str): Path to the PDF file.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        str: Comma-separated list of author names, or "Unknown authors" if not found.
    """
    logger.info(f"Extracting authors from {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        authors = "Unknown authors"
        author_lines = []

        # Step 1: Search the first page for author names
        first_page_text = doc[0].get_text()
        lines = first_page_text.split("\n")
        for i, line in enumerate(lines):
            # Match names in the format "First Last" or "First M. Last"
            if re.match(r"^[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+(?:∗)?$", line.strip()):
                author_lines.append(line.strip().replace("∗", ""))
            # Stop at email, abstract, or conference title
            elif re.match(r".*@\w+\.\w+", line) or line.strip().lower() in ["abstract", "keywords", "introduction"]:
                break
            elif re.search(r"^(?:[A-Z\s\d-]+|[A-Z][A-Za-z\s]+)\s*(?:Conference|Symposium|Workshop)", line, re.I):
                break
        if author_lines:
            authors = ", ".join(author_lines)
            logger.info(f"Authors extracted from name list (first page): {authors}")
            return authors

        # Step 2: Look for "By Author1, Author2, ..." pattern
        by_authors_pattern = r"By\s+([A-Za-z\s,]+(?:and [A-Za-z\s]+)?)(?=\n|$)"
        by_authors_match = re.search(by_authors_pattern, first_page_text)
        if by_authors_match:
            raw_authors = by_authors_match.group(1).strip()
            raw_authors = re.sub(r",\s*and\s*", ", ", raw_authors)
            authors = re.sub(r",\s*,", ",", raw_authors).strip(",").strip()
            logger.info(f"Authors extracted from 'By' line (first page): {authors}")
            return authors

        # Step 3: Look for "ACM Reference Format" pattern
        acm_ref_pattern = r"ACM Reference Format:\s*\n([A-Za-z\s,]+(?:and [A-Za-z\s]+)?)\.\s*\d{4}"
        acm_ref_match = re.search(acm_ref_pattern, first_page_text)
        if acm_ref_match:
            raw_authors = acm_ref_match.group(1).strip()
            raw_authors = re.sub(r",\s*and\s*", ", ", raw_authors)
            authors = re.sub(r",\s*,", ",", raw_authors).strip(",").strip()
            logger.info(f"Authors extracted from ACM Reference Format (first page): {authors}")
            return authors

        # Step 4: Search all pages if not found on the first page
        for page_num, page in enumerate(doc):
            if page_num == 0:  # Skip first page as it was already checked
                continue
            text = page.get_text()
            lines = text.split("\n")
            author_lines = []
            for i, line in enumerate(lines):
                if re.match(r"^[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+(?:∗)?$", line.strip()):
                    author_lines.append(line.strip().replace("∗", ""))
                elif re.match(r".*@\w+\.\w+", line) or line.strip().lower() in ["abstract", "keywords", "introduction"]:
                    break
                elif re.search(r"^(?:[A-Z\s\d-]+|[A-Z][A-Za-z\s]+)\s*(?:Conference|Symposium|Workshop)", line, re.I):
                    break
            if author_lines:
                authors = ", ".join(author_lines)
                logger.info(f"Authors extracted from name list (page {page_num + 1}): {authors}")
                return authors

            by_authors_match = re.search(by_authors_pattern, text)
            if by_authors_match:
                raw_authors = by_authors_match.group(1).strip()
                raw_authors = re.sub(r",\s*and\s*", ", ", raw_authors)
                authors = re.sub(r",\s*,", ",", raw_authors).strip(",").strip()
                logger.info(f"Authors extracted from 'By' line (page {page_num + 1}): {authors}")
                return authors

            acm_ref_match = re.search(acm_ref_pattern, text)
            if acm_ref_match:
                raw_authors = acm_ref_match.group(1).strip()
                raw_authors = re.sub(r",\s*and\s*", ", ", raw_authors)
                authors = re.sub(r",\s*,", ",", raw_authors).strip(",").strip()
                logger.info(f"Authors extracted from ACM Reference Format (page {page_num + 1}): {authors}")
                return authors

        # Step 5: Search for "Author Information" section (from last page backwards)
        for page_num in range(len(doc) - 1, -1, -1):
            text = doc[page_num].get_text()
            lines = text.split("\n")
            author_section_pattern = r"(?i)^(Author\s*Information|About\s*the\s*Authors?|Authors?)\s*$"
            for i, line in enumerate(lines):
                if re.match(author_section_pattern, line.strip()):
                    logger.info(f"Found 'Author Information' section on page {page_num + 1}")
                    author_lines = []
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if re.match(r"^[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+$", next_line):
                            author_lines.append(next_line)
                        elif re.match(r".*@\w+\.\w+", next_line) or next_line.lower() in ["references", "acknowledgments"]:
                            break
                        elif re.match(r"^[A-Za-z\s,]+(?:and [A-Za-z\s]+)?$", next_line):
                            raw_authors = next_line
                            raw_authors = re.sub(r",\s*and\s*", ", ", raw_authors)
                            authors = re.sub(r",\s*,", ",", raw_authors).strip(",").strip()
                            logger.info(f"Authors extracted from 'Author Information' section (page {page_num + 1}): {authors}")
                            return authors
                    if author_lines:
                        authors = ", ".join(author_lines)
                        logger.info(f"Authors extracted from 'Author Information' section (page {page_num + 1}): {authors}")
                        return authors
                    break

        # Step 6: Check PDF metadata for author information
        if authors == "Unknown authors":
            metadata = doc.metadata
            if "author" in metadata and metadata["author"]:
                authors = metadata["author"]
                logger.info(f"Authors extracted from metadata: {authors}")

        if authors == "Unknown authors":
            logger.warning(f"No authors found in {pdf_path}")

        return authors
    except Exception as e:
        logger.error(f"Error extracting authors from {pdf_path}: {str(e)}")
        return "Unknown authors"

# Load and preprocess PDF documents
def load_and_preprocess_pdf(pdf_path, logger, clean_latex, clean_text):
    """
    Load a PDF document and preprocess its content by cleaning LaTeX and normalizing text.
    
    Args:
        pdf_path (str): Path to the PDF file.
        logger (logging.Logger): Logger instance for logging messages.
        clean_latex (callable): Function to clean LaTeX symbols.
        clean_text (callable): Function to clean text.
    
    Returns:
        list: List of preprocessed documents.
    """
    logger.info("Loading document content with PyMuPDF")
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {pdf_path}")
    except Exception as e:
        logger.error(f"Error loading document content from {pdf_path}: {str(e)}")
        raise

    logger.info("Cleaning LaTeX and text")
    for doc in documents:
        doc.page_content = clean_latex(doc.page_content)
        doc.page_content = clean_text(doc.page_content)
    
    return documents