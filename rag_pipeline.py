import os, re, time, torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize embeddings for document vectorization
def initialize_embeddings(model_name, device, logger):
    """
    Initialize embeddings for document vectorization.
    
    Args:
        model_name (str): Name of the embedding model.
        device (str): Device to run the model on (e.g., 'mps', 'cpu').
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        HuggingFaceEmbeddings: Initialized embeddings.
    """
    logger.info("Initializing embeddings")
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

# Initialize the language model for answer generation
def initialize_llm(model_name, hf_token, device, max_new_tokens, logger):
    """
    Initialize the language model (LLM) for generating answers.
    
    Args:
        model_name (str): Name of the language model.
        hf_token (str): Hugging Face token for authentication.
        device (str): Device to run the model on (e.g., 'mps', 'cpu').
        max_new_tokens (int): Maximum number of tokens to generate.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        HuggingFacePipeline: Initialized LLM pipeline.
    """
    logger.info("Initializing LLM")
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

# Create or load a FAISS vector store for document retrieval
def create_or_load_vectorstore(documents, embeddings, pdf_name, chunk_size, chunk_overlap, logger):
    """
    Create or load a FAISS vector store for the given documents.
    
    Args:
        documents (list): List of documents to vectorize.
        embeddings (HuggingFaceEmbeddings): Embeddings for vectorization.
        pdf_name (str): Name of the PDF file (used for saving the index).
        chunk_size (int): Size of text chunks for splitting.
        chunk_overlap (int): Overlap between text chunks.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        FAISS: Vector store for document retrieval.
    """
    logger.info("Splitting text into chunks")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(texts)} chunks")
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        raise

    faiss_index_path = os.path.join(os.getcwd(), f"faiss_index_{pdf_name}")
    start_time = time.time()
    if os.path.exists(faiss_index_path):
        logger.info(f"FAISS index exists at {faiss_index_path}. Loading existing index...")
        try:
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FAISS index from {faiss_index_path}: {str(e)}")
            raise
    else:
        logger.info(f"Creating new vectorstore and saving to {faiss_index_path}...")
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(faiss_index_path)
            logger.info("FAISS index created and saved successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    load_time = time.time() - start_time
    logger.info(f"Vectorstore processing time: {load_time:.2f} seconds")
    return vectorstore

# Extract the final answer from the model response
def extract_answer(response, is_short_answer=False, context=""):
    """
    Extract the final answer from the model's response, removing duplicates and irrelevant details.
    
    Args:
        response (str): Raw response from the model.
        is_short_answer (bool): Whether to extract a short answer (not used in this version).
        context (str): Context for the answer (not used in this version).
    
    Returns:
        str: Cleaned and extracted answer.
    """
    match = re.search(r"Answer:(.*?)(?=\n|$)", response, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        sentences = answer.split(". ")
        seen_sentences = set()
        deduplicated_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (
                sentence
                and sentence not in seen_sentences
                and "specific details are not provided" not in sentence.lower()
            ):
                seen_sentences.add(sentence)
                deduplicated_sentences.append(sentence)
        answer = ". ".join(deduplicated_sentences)
        if not answer:
            return "Information not found in the context."
        return answer.strip() + ("." if not answer.endswith(".") else "")
    return "Information not found in the context."

# Create a RetrievalQA pipeline for answering questions
def create_rag_pipeline(llm, vectorstore, retriever_k, retriever_threshold, logger):
    """
    Create a RetrievalQA pipeline for answering questions using the RAG approach.
    
    Args:
        llm (HuggingFacePipeline): Language model for generating answers.
        vectorstore (FAISS): Vector store for document retrieval.
        retriever_k (int): Number of documents to retrieve.
        retriever_threshold (float): Similarity score threshold for retrieval.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        RetrievalQA: RAG pipeline for answering questions.
    """
    logger.info("Creating RAG pipeline")
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": retriever_k, "score_threshold": retriever_threshold},
            ),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    input_variables=["context", "question"],
                    template="You are an expert in analyzing academic research papers. Using ONLY the provided context, answer the question clearly and accurately, focusing on the most relevant information and avoiding repetition or irrelevant details. Provide a detailed answer with specific details (max 300 words), addressing both sides of an issue if applicable. If evidence is lacking, state 'Information not found in the context.' Do NOT invent information not present in the context or add speculative details such as assumed scales or methods not explicitly stated.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
                )
            },
        )
        logger.info("RAG pipeline created successfully")
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating RAG pipeline: {str(e)}")
        raise

# Generate answers for a list of questions using the RAG pipeline
def generate_answers(qa_chain, vectorstore, questions_list, doi, authors, retriever_k, retriever_threshold, remove_references, logger):
    """
    Generate answers for a list of questions using the RAG pipeline.
    
    Args:
        qa_chain (RetrievalQA): RAG pipeline for answering questions.
        vectorstore (FAISS): Vector store for document retrieval.
        questions_list (list): List of questions to answer.
        doi (str): DOI of the article.
        authors (str): Authors of the article.
        retriever_k (int): Number of documents to retrieve.
        retriever_threshold (float): Similarity score threshold for retrieval.
        remove_references (callable): Function to remove references from text.
        logger (logging.Logger): Logger instance for logging messages.
    
    Returns:
        dict: Dictionary mapping question features to answers.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": retriever_k, "score_threshold": retriever_threshold},
    )
    answers_dict = {}
    for q in questions_list:
        question = q["question"]
        is_short_answer = q["is_short_answer"]
        feature = q["feature"]

        logger.info(f"Processing question: {question}")

        if question == "What is the DOI of the article?" and doi:
            answer = f"The DOI of the article is https://doi.org/{doi}."
            logger.info(f"Answer (from extracted DOI): {answer}")
            answers_dict[feature] = answer
            continue
        elif question == "Who are the authors, and where and when was the article published?":
            answer = f"The authors are {authors}."
            logger.info(f"Answer (from extracted info): {answer}")
            answers_dict[feature] = answer
            continue

        logger.info("Retrieving documents")
        start_time = time.time()
        try:
            retrieved_docs = retriever.invoke(question)
            retrieval_time = time.time() - start_time
            logger.info(f"Retrieval time: {retrieval_time:.2f} seconds")
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Error retrieving documents for question '{question}': {str(e)}")
            answers_dict[feature] = "Error retrieving documents."
            continue

        retrieved_texts = []
        for doc in retrieved_docs:
            content = remove_references(doc.page_content)
            if content:
                retrieved_texts.append(content)
        combined_context = " ".join(retrieved_texts)

        logger.info("Generating answer")
        start_time = time.time()
        try:
            response = qa_chain.invoke(question)
            answer_time = time.time() - start_time
            logger.info(f"Answer generation time: {answer_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {str(e)}")
            answers_dict[feature] = "Error generating answer."
            continue

        answer_text = response["result"]
        answer = extract_answer(answer_text, is_short_answer)
        logger.info(f"Answer: {answer}")
        answers_dict[feature] = answer

    return answers_dict