# src/gia/core/utils.py
from __future__ import annotations
import os
import sys
import json
import time
import gc
import hashlib
import requests
import tomllib
from typing import Tuple, Optional, Dict, Any, Type, List
from datetime import datetime
from colorama import Fore, Style
import psutil
import GPUtil
import torch
from tqdm import tqdm
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex, StorageContext, PromptTemplate, SimpleDirectoryReader
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
)

# In utils.py
from typing import Optional
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from transformers import LogitsProcessor, LogitsProcessorList
from gptqmodel import GPTQModel
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openrouter import OpenRouter
from llama_index.llms.openai_like import OpenAILike

from llama_index.core import Settings

from huggingface_hub import HfApi
from gia.core.logger import logger, log_banner
from gia.config import CONFIG, get_config
from gia.core.state import state_manager
from gia.core.state import load_state
from gia.config import PROJECT_ROOT
import chromadb

log_banner(__file__)

CONTEXT_WINDOW = CONFIG["CONTEXT_WINDOW"]
MAX_NEW_TOKENS = CONFIG["MAX_NEW_TOKENS"]
TEMPERATURE = CONFIG["TEMPERATURE"]
TOP_P = CONFIG["TOP_P"]
TOP_K = CONFIG["TOP_K"]
REPETITION_PENALTY = CONFIG["REPETITION_PENALTY"]
NO_REPEAT_NGRAM_SIZE = CONFIG["NO_REPEAT_NGRAM_SIZE"]
MODEL_PATH = CONFIG["MODEL_PATH"]
EMBED_MODEL_PATH = CONFIG["EMBED_MODEL_PATH"]
DATA_PATH = CONFIG["DATA_PATH"]
DB_PATH = CONFIG["DB_PATH"]
DEBUG = CONFIG["DEBUG"]
DEVICE_MAP: str = ("cuda" if __import__("torch").cuda.is_available() else "cpu")

###

def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()

def append_to_chatbot(
    history: List[Dict[str, Any]],
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Append exactly one chat bubble; no shape filtering; safe dedupe on last assistant."""
    if not isinstance(history, list):
        raise ValueError("History must be a list of dicts.")
    if not isinstance(message, str):
        message = str(message)
    text = message.strip()
    if not text:
        return history

    role = (metadata or {}).get("role", "assistant")
    if role not in ("user", "assistant", "system"):
        role = "assistant"

    # Null-safe last-assistant dedupe
    if role == "assistant" and history and isinstance(history[-1], dict):
        last = history[-1]
        if last.get("role") == "assistant":
            last_meta = (last.get("metadata") or {})
            prev_hash = last_meta.get("_hash", "")
            cur_hash = _hash_text(text)
            if prev_hash == cur_hash:
                return history

    entry: Dict[str, Any] = {"role": role, "content": text}
    meta: Dict[str, Any] = {k: v for k, v in (metadata or {}).items() if k not in ("role", "options")}
    if role == "assistant":
        meta["_hash"] = _hash_text(text)
    if meta:
        entry["metadata"] = meta

    history.append(entry)
    return history


####

def get_qa_prompt() -> str:
    """
    Load QA prompt dynamically from config.toml (safe at import time).
    Appends "/no_think" if MODEL_TYPE == "qwen3".
    Returns a single string joined by newlines.
    """
    config_path = PROJECT_ROOT.parent.parent / "config.toml"
    qa_prompt_list: list[str] = []

    # Read prompt from config.toml (safe fallbacks)
    try:
        if config_path.exists():
            with config_path.open("rb") as f:
                _cfg = tomllib.load(f)
            prompt_data = _cfg.get("prompt", {}).get("qa_prompt", None)
            if isinstance(prompt_data, str):
                qa_prompt_list = [prompt_data]
            elif isinstance(prompt_data, list) and all(isinstance(item, str) for item in prompt_data):
                qa_prompt_list = prompt_data
            else:
                logger.info("Invalid 'prompt.qa_prompt' in config.toml (must be str or list[str]); using empty prompt.")
        else:
            logger.info(f"config.toml not found at {config_path}; using empty prompt.")
    except (tomllib.TOMLDecodeError, OSError) as e:
        logger.info(f"Error reading/parsing config.toml: {e}; using empty prompt.")
    except Exception as e:
        logger.info(f"Unexpected error in get_qa_prompt: {e}; using empty prompt.")

    # Import-time safe MODEL_TYPE resolution
    model_type = None
    try:
        # Only call state_manager.get_state if it's actually available & callable
        get_state = getattr(state_manager, "get_state", None)
        if callable(get_state):
            model_type = get_state("MODEL_TYPE", CONFIG.get("MODEL_TYPE"))
        else:
            model_type = CONFIG.get("MODEL_TYPE")
    except Exception as e:
        logger.debug(f"(STATE) get_state unavailable during import: {e}; defaulting to CONFIG['MODEL_TYPE'].")
        model_type = CONFIG.get("MODEL_TYPE")

    if isinstance(model_type, str) and model_type.lower() == "qwen3":
        qa_prompt_list.append("/no_think")

    return "\n".join(qa_prompt_list)
QA_PROMPT_TMPL = get_qa_prompt()
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

####


def get_system_info() -> tuple[float, float, list[float]]:
    """Retrieve system resource usage (CPU, memory, GPU).

    Returns:
        Tuple of CPU usage (%), memory usage (%), and GPU usage (% list).
    """
    # TO GET CPU USAGE
    cpu_usage = psutil.cpu_percent(interval=1)
    # TO GET MEMORY USAGE
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    # TO GET GPU USAGE WITH FALLBACK
    gpu_usage = []
    try:
        gpus = GPUtil.getGPUs()
        gpu_usage = [gpu.load * 100 for gpu in gpus]
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to get GPU usage (GPUtil unavailable): {e}")
        gpu_usage = [0.0]  # FALLBACK: Assume no GPU usage
    return cpu_usage, memory_usage, gpu_usage


####

# PRESENCE PENALTY PROCESSOR - FOR CLEANER OUTPUT
class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Custom logits processor for presence penalty."""

    def __init__(self, presence_penalty: float = 1.0):
        # TO SET PENALTY
        self.presence_penalty = presence_penalty

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # TO APPLY PENALTY ON GPU
        batch_size, vocab_size = scores.shape
        presence = torch.zeros_like(scores)
        for b in range(batch_size):
            unique_tokens = torch.unique(input_ids[b])
            valid_mask = unique_tokens < vocab_size
            unique_tokens = unique_tokens[valid_mask]
            presence[b, unique_tokens] = self.presence_penalty
        return scores - presence

global_logits_processor = LogitsProcessorList([PresencePenaltyLogitsProcessor()])

####

def _init_embed_model(device: str) -> HuggingFaceEmbedding:
    """Initialize embedding model with shared config.

    Args:
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        HuggingFaceEmbedding: Configured embedding model.
    """
    # INITIALIZE EMBEDDING MODEL WITH SHARED CONFIG
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        max_length=512,
        normalize=True,
        embed_batch_size=256,
        device=device,
    )
    if device == "cuda":
        embed_model._model = embed_model._model.half()  # FP16 FOR VRAM EFFICIENCY
    return embed_model

#DB CREATION WITH OPTIMIZED EMBEDDINGS
def _resolve_collection_name() -> str:
    """
    Resolve the Chroma collection name from config.toml with safe fallbacks.
    Order of precedence:
      1) [database].collection_name in config.toml
      2) CONFIG["COLLECTION_NAME"] if set
      3) hard default "GIA_db"
    Validates length (3..63). Never returns None.
    """
    # 1) Read config.toml directly for runtime truth
    cfg_path = PROJECT_ROOT.parent.parent / "config.toml"
    name_from_toml = None
    try:
        if cfg_path.exists():
            with cfg_path.open("rb") as f:
                _cfg = tomllib.load(f)
            name_from_toml = (
                _cfg.get("database", {}).get("collection_name", None)
                if isinstance(_cfg, dict) else None
            )
    except Exception as e:
        logger.warning(f"(CFG) Failed reading config.toml for collection_name: {e}")

    # 2) Fallbacks: CONFIG then hard default
    name = (
        name_from_toml
        or CONFIG.get("COLLECTION_NAME")
        or "GIA_db"
    )

    # 3) Validate & sanitize
    if not isinstance(name, str):
        logger.warning("(CFG) COLLECTION_NAME not a string; using 'GIA_db'")
        name = "GIA_db"
    n = len(name)
    if n < 3 or n > 63:
        logger.warning("(CFG) COLLECTION_NAME length %d invalid; using 'GIA_db'", n)
        name = "GIA_db"

    return name


def save_database():
    """Build and persist ChromaDB database with optimized embeddings."""
    state = load_state()  # DOT-ACCESS SNAPSHOT
    if state.DATABASE_LOADED:
        logger.info("(UDB) Database already loaded, skipping save")
        return
    start_time = time.time()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        torch.cuda.empty_cache()

        # Ensure DB path exists
        Path(DB_PATH).mkdir(parents=True, exist_ok=True)

        # Authoritative collection name (config.toml -> CONFIG -> 'GIA_db')
        collection_name = _resolve_collection_name()
        logger.info(f"(UDB) Using Chroma collection: {collection_name}")

        # Initialize ChromaDB client & collection
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Embedding model
        embed_model = _init_embed_model(device)

        # Empty index (we insert nodes incrementally)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=[],
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=False,
        )

        # Load *.txt files (graceful if none)
        if not os.path.isdir(DATA_PATH):
            logger.warning(f"DATA_PATH does not exist: {DATA_PATH} — proceeding with empty index.")
            file_paths: list[str] = []
        else:
            file_paths = [
                os.path.join(DATA_PATH, f)
                for f in os.listdir(DATA_PATH)
                if f.endswith(".txt")
            ]
        total_files = len(file_paths)
        logger.info(f"Found {total_files} text files to process")

        file_batch_size = 400
        chunk_size = 512
        chunk_overlap = 50
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for batch_start in tqdm(range(0, total_files, file_batch_size), desc="Processing file batches"):
            batch_end = min(batch_start + file_batch_size, total_files)
            batch_paths = file_paths[batch_start:batch_end]
            try:
                logger.info(f"Loading batch {batch_start // file_batch_size + 1} ({batch_start}-{batch_end})")
                documents = SimpleDirectoryReader(input_files=batch_paths, encoding="utf-8").load_data()
                logger.info(f"Loaded {len(documents)} documents in batch")

                batch_nodes = []
                for doc in documents:
                    chunks = splitter.split_text(doc.text)
                    for chunk in chunks:
                        node_metadata = {
                            "filename": os.path.basename(doc.metadata.get("file_path", "")),
                            "date": time.strftime("%Y-%m-%d"),
                        }
                        batch_nodes.append(TextNode(text=chunk, metadata=node_metadata))

                logger.info(f"Created {len(batch_nodes)} nodes in batch")
                sub_batch_size = 100
                for sub_start in range(0, len(batch_nodes), sub_batch_size):
                    sub_batch = batch_nodes[sub_start : sub_start + sub_batch_size]
                    index.insert_nodes(sub_batch)
                    torch.cuda.empty_cache()

                logger.info(f"Completed batch: {len(batch_nodes)} nodes inserted")
                logger.info(f"Current node count in ChromaDB: {chroma_collection.count()}")
                del documents, batch_nodes
            except Exception as e:
                logger.error(f"Error in batch {batch_start // file_batch_size + 1}: {e}")
                torch.cuda.empty_cache()
                continue

        elapsed_time = time.time() - start_time
        logger.info(
            f"ChromaDB database with ~{total_files} files (~{chroma_collection.count()} nodes) created in: {elapsed_time / 3600:.2f} hours"
        )

        # Persist handles in state
        state_manager.set_state("CHROMA_COLLECTION", chroma_collection)
        state_manager.set_state("EMBED_MODEL", embed_model)
        state_manager.set_state("INDEX", index)
        state_manager.set_state("DATABASE_LOADED", True)
    except Exception as e:
        logger.critical(f"Database creation failed: {e}")
        raise


####

def load_database(llm: Optional[object] = None) -> str:
    """One-time ChromaDB initializer."""
    state = load_state()
    if state.DATABASE_LOADED:
        logger.info("(UDB) Database already loaded, skipping initialization")
        logger.debug(f"(UDB) Index state: {state.INDEX}")
        return "Database already loaded.\n"

    try:
        logger.info("(UDB) Initializing database")
        clear_vram()
        start_time = time.time()

        # Ensure DB path exists
        Path(DB_PATH).mkdir(parents=True, exist_ok=True)

        # Authoritative collection name
        collection_name = _resolve_collection_name()
        logger.info(f"(UDB) Using Chroma collection: {collection_name}")

        # Initialize ChromaDB client & collection
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_model = _init_embed_model(device)

        # Vector store & index
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.debug(f"(UDB) Index set: {index}")

        if llm is None:
            raise ValueError("Error: no model loaded. Confirm a model before loading the database.")

        # Query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=2,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.65)],
            vector_store_query_mode="hybrid",
            response_mode="compact",
            text_qa_template=QA_PROMPT,
            streaming=True,
            verbose=False,
        )
        logger.debug(f"(UDB) Query engine set: {query_engine}")

        logger.info(f"ChromaDB loaded with {chroma_collection.count()} entries in {time.time() - start_time:.2f}s")
        clear_vram()

        # Persist handles in state
        state_manager.set_state("CHROMA_COLLECTION", chroma_collection)
        state_manager.set_state("EMBED_MODEL", embed_model)
        state_manager.set_state("INDEX", index)
        state_manager.set_state("QUERY_ENGINE", query_engine)
        state_manager.set_state("DATABASE_LOADED", True)
        return "Database loaded. Query engine is ready.\n"
    except Exception as e:
        logger.critical(f"Database loading failed: {e}")
        state_manager.set_state("INDEX", None)
        raise


####

def update_database(
    filepaths: list[str] | str,
    query: str,
    metadata: Optional[dict] = None,
    embed_model: Optional[object] = None,
    llm: Optional[object] = None,
) -> None:
    state = load_state()  # LOAD STATE AT START FOR DOT ACCESS
    filepaths = [filepaths] if isinstance(filepaths, str) else filepaths
    if not all(Path(fp).exists() for fp in filepaths):
        raise FileNotFoundError(f"One or more files not found: {filepaths}")

    if state.CHROMA_COLLECTION is None:
        Path(DB_PATH).mkdir(parents=True, exist_ok=True)
        db = chromadb.PersistentClient(path=DB_PATH)
        chroma_collection = db.get_or_create_collection(
            _resolve_collection_name(),
            metadata={"hnsw:space": "cosine"},
        )
    else:
        chroma_collection = state.CHROMA_COLLECTION

    model = embed_model or state.EMBED_MODEL
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _init_embed_model(device)

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    for filepath in filepaths:
        text = Path(filepath).read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        if not chunks:
            logger.warning(f"No chunks created for {filepath}")
            continue

        try:
            embeddings = model.get_text_embedding_batch(chunks, show_progress=False)
        except Exception as exc:
            logger.error(f"Embedding failed for {filepath}: {exc}")
            continue

        base_meta = (metadata or {}).copy()
        if "tags" in base_meta:
            if isinstance(base_meta["tags"], str):
                base_meta["tags"] = [
                    t.strip().lower() for t in base_meta["tags"].split(",") if t.strip()
                ]
            elif isinstance(base_meta["tags"], list):
                base_meta["tags"] = [
                    t.lower().strip()
                    for t in base_meta["tags"]
                    if isinstance(t, str) and t.strip()
                ]
            else:
                base_meta["tags"] = []  # FORCE LIST ON INVALID TYPE
        base_meta["filename"] = os.path.basename(filepath)
        if isinstance(base_meta.get("sources"), list):
            base_meta["sources"] = ", ".join(str(s) for s in base_meta["sources"])

        chunk_metadatas = [base_meta.copy() for _ in chunks]
        doc_id = hashlib.md5(filepath.encode("utf-8")).hexdigest()
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        try:
            chroma_collection.upsert(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
            )
            logger.info(
                f"Upserted {len(chunks)} chunks for {filepath}. "
                f"Collection count: {chroma_collection.count()}"
            )
        except Exception as exc:
            logger.error(f"Upsert failed for {filepath}: {exc}")
    # Set state after initialization
    state_manager.set_state("CHROMA_COLLECTION", chroma_collection)
    state_manager.set_state("DATABASE_LOADED", True)
    state_manager.set_state("EMBED_MODEL", model)


def unload_model() -> str:
    """Unload the current model and clear resources.

    Returns:
        str: Status message.
    """
    # TO RETRIEVE CURRENT LLM FROM STATE
    llm = state_manager.get_state("LLM")
    if llm is None:
        return "No model loaded to unload."
    try:
        # TO HANDLE HUGGINGFACE-SPECIFIC UNLOAD
        if isinstance(llm, HuggingFaceLLM):
            if hasattr(llm, "_model"):
                del llm._model
            if hasattr(llm, "_tokenizer"):
                del llm._tokenizer
            del llm
            clear_vram()
        else:
            # TO HANDLE ONLINE/OTHER LLMS
            del llm
            gc.collect()
        # TO RESET STATE KEYS
        state_manager.set_state("LLM", None)
        state_manager.set_state("MODEL_NAME", None)
        state_manager.set_state("MODEL_PATH", None)
        return "Model unloaded successfully."
    except Exception as e:
        logger.error(f"Unload failed: {e}")
        return f"Error unloading model: {str(e)}"

####

def filtered_query_engine(
    llm,
    query_str: str,
    category: str,
) -> RetrieverQueryEngine:
    state = load_state()  # LOAD STATE AT START FOR DOT ACCESS
    """
    Category-restricted query engine with no post-retrieval chunking.
    """


    if state_manager.get_state("INDEX") is None:
        logger.critical("Index not initialized")
        raise ValueError("Index must be initialized")
    if not query_str:
        logger.critical("Query string is empty")
        raise ValueError("query_str must be provided")
    if not category:
        logger.critical("Category is empty")
        raise ValueError("category must be provided")

    logger.debug("(UDB) filtered_query_engine | category=%s", category)

    try:

        meta_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value=category.lower(),
                    operator=FilterOperator.EQ,
                )
            ]
        )


        retriever = state_manager.get_state("INDEX").as_retriever(
            filters=meta_filters,
            similarity_top_k=2,
            verbose=logger.isEnabledFor(logger.DEBUG),
        )


        synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="compact",  # No chunking; independent per-node
            text_qa_template=QA_PROMPT,
        )


        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.65)],
        )

        logger.info("(UDB) Filtered query engine created")
        return query_engine

    except torch.cuda.OutOfMemoryError as e:
        logger.warning(
            f"OOM in query engine: {e}. Clearing cache and retrying with top_k=1."
        )
        clear_vram()
        retriever = state_manager.get_state("INDEX").as_retriever(
            filters=meta_filters,
            similarity_top_k=1,
            verbose=logger.isEnabledFor(logger.DEBUG),
        )
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.65)],
        )
        return query_engine
    except Exception as exc:
        logger.exception("(UDB) filtered_query_engine FAILED: %s", exc)
        raise RuntimeError(f"Query engine creation failed: {exc}") from exc

###

def clear_vram() -> None:
    """Clear VRAM with retry loop for CUDA errors."""
    original_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    retries = 3
    for attempt in range(retries):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            cpu_usage, memory_usage, gpu_usage = get_system_info()  # FIXED UNPACKING
            if memory_usage > 80 or any(g > 80 for g in gpu_usage):
                logger.warning(
                    f"High usage: CPU {cpu_usage}%, RAM {memory_usage}%, GPU {gpu_usage}%—cleared, monitor for leaks."
                )
            logger.info(
                f"VRAM cleared—CPU: {cpu_usage}%, RAM: {memory_usage}%, GPU: {gpu_usage}%"
            )
            return
        except RuntimeError as e:
            if "unknown error" in str(e) and attempt < retries - 1:
                logger.warning(
                    f"CUDA unknown error on attempt {attempt+1}: {e}. Retrying..."
                )
                time.sleep(1)  # SHORT PAUSE TO LET KERNELS SETTLE
                continue
            else:
                logger.error(f"VRAM clear failed after {retries} attempts: {e}")
                raise
        finally:
            torch.backends.cudnn.benchmark = original_benchmark

def clear_ram(
    threshold=80,
): 
    gc.collect()  
    mem = psutil.virtual_memory().percent
    if mem > threshold and os.geteuid() == 0:  # ROOT-ONLY FOR DROP_CACHES; SKIP IF NOT
        try:
            os.system(
                "sync; echo 3 > /proc/sys/vm/drop_caches"
            ) 
            logger.info(f"System RAM cleared (was {mem}% > {threshold}%)")
        except Exception as e:
            logger.warning(f"RAM clear failed: {e}. Need sudo?")
    else:
        logger.debug(f"RAM at {mem}% < {threshold}%; skipped clear")

def fetch_openai_models() -> list:
    """Fetch available OpenAI models from the API for dropdown choices, only valid chat/completion models."""
    valid_models = [
        "o1", "o1-2024-12-17", "o1-pro", "o1-pro-2025-03-19", "o1-preview", "o1-preview-2024-09-12", "o1-mini", "o1-mini-2024-09-12",
        "o3-mini", "o3-mini-2025-01-31", "o3", "o3-2025-04-16", "o3-pro", "o3-pro-2025-06-10", "o4-mini", "o4-mini-2025-04-16",
        "gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-vision-preview", "gpt-4-1106-vision-preview",
        "gpt-4-turbo-2024-04-09", "gpt-4-turbo", "gpt-4o", "gpt-4o-audio-preview", "gpt-4o-audio-preview-2024-12-17", "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-mini-audio-preview", "gpt-4o-mini-audio-preview-2024-12-17", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
        "gpt-4.5-preview", "gpt-4.5-preview-2025-02-27", "chatgpt-4o-latest", "gpt-4o-mini", "gpt-4o-mini-2024-07-18", "gpt-4-0613",
        "gpt-4-32k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301", "text-davinci-003", "text-davinci-002", "gpt-3.5-turbo-instruct",
        "text-ada-001", "text-babbage-001", "text-curie-001", "ada", "babbage", "curie", "davinci", "gpt-35-turbo-16k", "gpt-35-turbo",
        "gpt-35-turbo-0125", "gpt-35-turbo-1106", "gpt-35-turbo-0613", "gpt-35-turbo-16k-0613"
    ]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return valid_models
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Only include valid models present in the API response
        available = set(m["id"] for m in data.get("data", []))
        filtered = [m for m in valid_models if m in available]
        return filtered or valid_models
    except Exception as e:
        logger.warning(f"Could not fetch OpenAI models for dropdown: {e}")
        return valid_models

def load_llm(mode: str, config: Dict) -> Tuple[str, Optional[LLM]]:
    """Load LLM based on mode and config.
    Args:
        mode: LLM mode.
        config: Configuration dict.
    Returns:
        Message and loaded LLM or None.
    """
    # TO SAFELY EXTRACT MODEL NAME - FALLBACK TO CONFIG OR GLOBAL; SECURITY: NO EXECUTION, STRING ONLY.
    try:
        logger.debug(f"Loading LLM for mode: {mode} with config: {config}")
        model_name = config.get("model_name") or config.get("model_path") or MODEL_PATH
        llm = create_llm(mode, model_name)
        clear_vram()
        logger.debug(f"Load success for {mode}: {model_name}")
        return f"{mode} model loaded: {model_name}", llm
    except Exception as e:
        # TO HANDLE LOAD ERRORS GRACEFULLY - LOG WITHOUT EXPOSING SENSITIVE DETAILS; RISK: PARTIAL STATE IF FAILS, BUT CALLER HANDLES NONE.
        error_msg = f"Load failed for {mode}: {str(e)}"
        logger.error(error_msg)
        return error_msg, None
    
from transformers import AutoTokenizer
def create_llm(mode: str, model_name: str) -> Optional[LLM]:
    """Create LLM instance based on mode and model name."""
    # TO VALIDATE MODE - PREVENTS INVALID OPERATIONS EARLY; SECURITY: RESTRICTS TO KNOWN MODES, NO ARBITRARY EXEC.
    if mode not in {"Local", "HuggingFace", "OpenAI", "OpenRouter"}:
        raise ValueError("Invalid mode. Use: Local | HuggingFace | OpenAI | OpenRouter.")
    # TO VALIDATE MODEL NAME - ENSURES NON-EMPTY STRING; EDGE CASE: EMPTY LEADS TO API FAILURES DOWNSTREAM.
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty string.")
    # TO PRE-CLEAR VRAM - ENSURES RESOURCE AVAILABILITY; PERFORMANCE: PREVENTS OOM IN LOAD; RISK: IF FAILS, SUBSEQUENT CALLS MAY STILL OOM BUT HANDLED IN EXCEPT.
    clear_vram()
    logger.debug(f"Creating LLM for mode: {mode}, model: {model_name}")
    if mode == "Local":
        # TO SET EXPLICIT MODE - ENABLES ACCURATE DETECTION ACROSS THREADS; ASSUMPTION: LOCAL IMPLIES OFFLINE/QUANTIZED MODEL.
        state_manager.set_state("MODE", "Local")
        # TO LOAD QUANTIZED MODEL - USES CORRECT API PER LIBRARY DOCS; SECURITY: TRUST_REMOTE_CODE=TRUE FOR HF COMPAT, BUT LIMIT TO TRUSTED MODELS; EDGE CASE: INVALID PATH RAISES, CAUGHT BELOW.
        pretrained = GPTQModel.from_quantized(
            model_name,
            device_map=DEVICE_MAP,
            use_safetensors=True,
            trust_remote_code=True,
        )
        llm = HuggingFaceLLM(
            model=pretrained.model,
            tokenizer=pretrained.tokenizer,
            context_window=int(CONTEXT_WINDOW),
            system_prompt="",
            query_wrapper_prompt="",
            generate_kwargs={
                "temperature": float(TEMPERATURE),
                "top_p": float(TOP_P),
                "top_k": int(TOP_K),
                "repetition_penalty": float(REPETITION_PENALTY),
                "no_repeat_ngram_size": int(NO_REPEAT_NGRAM_SIZE),
                "do_sample": True,
                "logits_processor": global_logits_processor,
            },
            device_map=DEVICE_MAP,
            is_chat_model=False,
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("HF Local LLM ready (no cap in generate_kwargs; QE/direct compatible).")
        return llm
    if mode == "HuggingFace":
        # TO SET EXPLICIT MODE - INDICATES ONLINE/API USAGE; PERFORMANCE: NO LOCAL RESOURCES NEEDED.
        state_manager.set_state("MODE", "Online")
        llm = HuggingFaceInferenceAPI(
            model_name=model_name,
            api_key=__import__("os").getenv("HF_TOKEN"),
            provider="auto",
            is_chat_model=False,
            num_output=int(MAX_NEW_TOKENS),       # SINGLE SOURCE OF TRUTH FOR OUTPUT CAP
            temperature=float(TEMPERATURE),       # SAFE SAMPLING PARAMS
            top_p=float(TOP_P),
        )
        token = __import__("os").getenv("HF_TOKEN")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            llm.__pydantic_private__["_tokenizer"] = tokenizer
            logger.debug(f"Tokenizer loaded and attached to __pydantic_private__ for {model_name}.")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {str(e)}. Token counting may be approximate.")
        state_manager.set_state("USE_CHAT", False)
        logger.info("HF Inference API LLM ready (single cap via constructor; chat disabled).")
        return llm
    if mode == "OpenAI":
        # TO SET EXPLICIT MODE - INDICATES ONLINE/API USAGE; SECURITY: API KEY FROM ENV, NO HARDCODE.
        state_manager.set_state("MODE", "Online")
        llm = LlamaOpenAI(
            model=model_name,
            api_key=__import__("os").getenv("OPENAI_API_KEY"),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            max_tokens=int(MAX_NEW_TOKENS),        # SINGLE CAP VIA PROVIDER ARG
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("OpenAI LLM ready (cap via max_tokens in constructor).")
        return llm
    # OPENROUTER — STREAMS FINE; USE max_tokens IN CTOR
    if mode == "OpenRouter":
        # TO SET EXPLICIT MODE - INDICATES ONLINE/API USAGE; EDGE CASE: MISSING KEY RAISES IN CONSTRUCTOR.
        state_manager.set_state("MODE", "Online")
        llm = OpenRouter(
            model=model_name,
            api_key=__import__("os").getenv("OPENROUTER_API_KEY"),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            max_tokens=int(MAX_NEW_TOKENS),        # SINGLE CAP VIA PROVIDER ARG
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("OpenRouter LLM ready (cap via max_tokens in constructor).")
        return llm
    # UNREACHABLE DUE TO VALIDATION
    raise ValueError(f"Unhandled mode: {mode}")

def generate(
    query: str,
    system_prompt: str,
    llm: Any,
    max_new_tokens: Optional[int] = None,
    think: bool = False,
    depth: int = 0,
    **kwargs: Any,
) -> str:
    """
    PRIMARY GENERATION PIPELINE (NON-STREAM). BULLETPROOF FOR TRANSFORMERS >= 4.55.

    Args:
        query: User prompt (free-form).
        system_prompt: System instruction prefix.
        llm: LlamaIndex HuggingFaceLLM or OpenAI-like LLM wrapper.
        max_new_tokens: Optional hard cap for generation length.
        think: Reserved flag (kept for interface compatibility).
        depth: Retry depth (for OOM exponential backoff).
        **kwargs: Extra model.generate() kwargs (safe subset only).

    Returns:
        Model text output (str). Returns a descriptive error message on failure.
    """

    start_time = time.time()

    # === INPUT SANITY & PROMPT ASSEMBLY ======================================
    user_text = (query or "").strip()
    sys_text = (system_prompt or "").strip()
    if not user_text and not sys_text:
        return "EMPTY PROMPT."

    prompt = f"{sys_text}\n{user_text}" if sys_text else user_text

    # === TARGET TOKENS & STATE SNAPSHOT ======================================
    previous_max: Optional[int] = getattr(llm, "max_new_tokens", None)
    target_tokens: int = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else int(previous_max or 256)
    )

    try:
        # === HUGGINGFACE (LLAMA-INDEX WRAPPER) PATH ==========================
        try:
            from llama_index.llms.huggingface import HuggingFaceLLM as _HFL_TYPE  # type: ignore[attr-defined]
        except Exception:
            _HFL_TYPE = tuple()  # type: ignore[assignment]

        if isinstance(llm, _HFL_TYPE) and hasattr(llm, "_model") and hasattr(llm, "_tokenizer"):
            model = llm._model
            tok = llm._tokenizer
            device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # --- TOKENIZE WITH SPECIALS; AVOID ZERO-LENGTH PROMPTS -----------
            enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.get("input_ids", None)

            # # MILITARY GUARD: ENSURE SEQ_LEN >= 1 (BOS -> EOS -> 0)
            if input_ids is None or input_ids.numel() == 0 or input_ids.shape[-1] == 0:
                if getattr(tok, "bos_token_id", None) is not None:
                    fid = tok.bos_token_id
                elif getattr(tok, "eos_token_id", None) is not None:
                    fid = tok.eos_token_id
                else:
                    fid = 0  # LAST RESORT: NON-NEGATIVE TOKEN INDEX
                    logger.warning("No BOS/EOS; using token_id=0 as fallback to avoid empty input_ids.")
                input_ids = torch.tensor([[fid]], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                enc = {"input_ids": input_ids, "attention_mask": attention_mask}

            # DEVICE MOVE (DICT-SAFE) -----------------------------------------
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is None:
                # ENSURE A VALID MASK IF TOKENIZER DID NOT PROVIDE ONE
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
            else:
                attention_mask = attention_mask.to(device)

            # PAD-ID FALLBACK (SUPPRESSES HF PAD WARNINGS) --------------------
            pad_id: Optional[int] = getattr(tok, "pad_token_id", None)
            if pad_id is None and getattr(tok, "eos_token_id", None) is not None:
                pad_id = tok.eos_token_id  # pass as kwarg (do not mutate tok globally)

            # --- GENERATION KWARGS (MINIMAL, EXPLICIT) ------------------------
            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": target_tokens,
                "do_sample": True,
                "top_p": float(getattr(llm, "top_p", kwargs.get("top_p", 0.95))),
                "temperature": float(getattr(llm, "temperature", kwargs.get("temperature", 0.7))),
                "use_cache": False,  # <-- KEY FIX: BYPASS cache_position PATH
                "pad_token_id": pad_id,
                "attention_mask": attention_mask,
                "return_dict_in_generate": True,
            }
            # MERGE SAFE EXTRAS WITHOUT OVERRIDING CRITICAL GUARDS ------------
            for k in ("repetition_penalty", "top_k", "min_new_tokens", "no_repeat_ngram_size"):
                if k in kwargs and kwargs[k] is not None:
                    gen_kwargs[k] = kwargs[k]

            # FINAL ASSERT: NO ZERO-LENGTH input_ids --------------------------
            if gen_kwargs["input_ids"].shape[-1] == 0:
                raise RuntimeError("Guard failed: input_ids has zero length before generate().")

            # --- RUN GENERATION (SYNC) ----------------------------------------
            with torch.inference_mode():
                out = model.generate(**gen_kwargs)

            # --- DECODE: SLICE OFF PROMPT, SKIP SPECIALS ---------------------
            prompt_len = input_ids.shape[-1]
            # out may be dict-like (return_dict_in_generate=True)
            sequences = out.sequences if hasattr(out, "sequences") else out
            new_tokens = sequences[:, prompt_len:] if sequences.shape[-1] > prompt_len else sequences
            text = tok.decode(new_tokens[0], skip_special_tokens=True)

            # === ACCOUNTING / METRICS (BEST-EFFORT) ==========================
            streamed_text = text or ""
            if not streamed_text.strip():
                logger.warning("EMPTY RESPONSE — CHECK PROMPT OR TOKEN LIMITS.")
                return "EMPTY RESPONSE FROM MODEL."

            # TOKEN USAGE ACCOUNTING (PREFERS TIKTOKEN, FALLBACK WORD SPLIT)
            try:
                import tiktoken  # type: ignore
                encoding = tiktoken.get_encoding("cl100k_base")
                total_tokens = len(encoding.encode(streamed_text))
            except Exception as e:
                logger.warning(f"TIKTOKEN FAILED: {e} — FALLBACK TO WORD COUNT.")
                total_tokens = len(streamed_text.split())

            # PERF METRICS LOGGING
            duration = time.time() - start_time
            tokens_per_sec = total_tokens / duration if duration > 0 else 0
            try:
                run_count = (state_manager.get_state("run_count") or 0) + 1  # type: ignore[name-defined]
                total_tokens_all = (state_manager.get_state("total_tokens_all") or 0) + total_tokens  # type: ignore[name-defined]
                total_duration_all = (state_manager.get_state("total_duration_all") or 0.0) + duration  # type: ignore[name-defined]
                state_manager.set_state("run_count", run_count)  # type: ignore[name-defined]
                state_manager.set_state("total_tokens_all", total_tokens_all)  # type: ignore[name-defined]
                state_manager.set_state("total_duration_all", total_duration_all)  # type: ignore[name-defined]
                avg_tps = total_tokens_all / total_duration_all if total_duration_all > 0 else 0
                logger.info(
                    f"INFER TIME={duration:.2f}s | TOKENS={total_tokens} | "
                    f"AVG_TPS={avg_tps:.2f} | CURR_TPS={tokens_per_sec:.2f}"
                )
            except Exception:
                # STATE MANAGER OPTIONAL — DO NOT HARD FAIL ON METRICS
                pass

            return streamed_text.strip()

        # === OPENAI-LIKE PATH (FALLBACK) =====================================
        complete = getattr(llm, "complete", None)
        if callable(complete):
            resp = llm.complete(f"{system_prompt}\n{query}")
            return (resp.text if hasattr(resp, "text") else str(resp)).strip()

        return "LLM BACKEND UNSUPPORTED FOR GENERATION."

    # === OOM HANDLING: REDUCE TOKENS AND RECUR ===============================
    except torch.cuda.OutOfMemoryError as e:
        logger.warning(f"OOM DURING GENERATION — {e}. RETRYING WITH HALF TOKENS.")
        clear_vram()
        new_max = max(1, (target_tokens // 2))
        if hasattr(llm, "max_new_tokens"):
            llm.max_new_tokens = new_max
        return generate(
            query,
            system_prompt,
            llm,
            max_new_tokens=new_max,
            think=think,
            depth=depth + 1,
            **kwargs,
        )

    # === GENERIC FAILURE =====================================================
    except Exception as e:
        error_msg = f"GENERATION FAILURE — {e}"
        logger.error(error_msg)
        return error_msg

    # === ALWAYS CLEAR VRAM + RESTORE ORIGINAL STATE ==========================
    finally:
        clear_vram()
        if max_new_tokens is not None and previous_max is not None and hasattr(llm, "max_new_tokens"):
            llm.max_new_tokens = previous_max



__all__ = [
    "generate",
    "update_database",
    "clear_vram",
    "save_database",
    "load_database",
    "get_system_info",
    "append_to_chatbot",
    "PresencePenaltyLogitsProcessor", 
    "filtered_query_engine",
    "clear_ram",
    "fetch_openai_models",
    "create_llm",
    "load_llm",
    "unload_model",
]
