# src/gia/core/utils.py
from __future__ import annotations
import os
import time
import gc
import hashlib
import requests
import tomllib
from typing import Tuple, Optional, Dict, Any, List, Iterable, Generator, Callable
import tiktoken
import psutil
import GPUtil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    PromptTemplate,
    SimpleDirectoryReader,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from gptqmodel import GPTQModel, QuantizeConfig
from llama_index.llms.openrouter import OpenRouter
from llama_index.readers.file import PyPDFReader

from gia.core.logger import logger, log_banner
from gia.config import CONFIG, system_prefix, PROJECT_ROOT
from gia.core.state import state_manager, load_state
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
DEVICE_MAP: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


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
            last_meta = last.get("metadata") or {}
            prev_hash = last_meta.get("_hash", "")
            cur_hash = _hash_text(text)
            if prev_hash == cur_hash:
                return history

    entry: Dict[str, Any] = {"role": role, "content": text}
    meta: Dict[str, Any] = {
        k: v for k, v in (metadata or {}).items() if k not in ("role", "options")
    }
    if role == "assistant":
        meta["_hash"] = _hash_text(text)
    if meta:
        entry["metadata"] = meta

    history.append(entry)
    return history


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
            elif isinstance(prompt_data, list) and all(
                isinstance(item, str) for item in prompt_data
            ):
                qa_prompt_list = prompt_data
            else:
                logger.info(
                    "Invalid 'prompt.qa_prompt' in config.toml (must be str or list[str]); using empty prompt."
                )
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
        logger.debug(
            f"(STATE) get_state unavailable during import: {e}; defaulting to CONFIG['MODEL_TYPE']."
        )
        model_type = CONFIG.get("MODEL_TYPE")

    if isinstance(model_type, str) and model_type.lower() == "qwen3":
        qa_prompt_list.append("/no_think")

    return "\n".join(qa_prompt_list)


QA_PROMPT_TMPL = get_qa_prompt()
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


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
                if isinstance(_cfg, dict)
                else None
            )
    except Exception as e:
        logger.warning(f"(CFG) Failed reading config.toml for collection_name: {e}")

    # 2) Fallbacks: CONFIG then hard default
    name = name_from_toml or CONFIG.get("COLLECTION_NAME") or "GIA_db"

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
    """Build and persist ChromaDB with optimized embeddings from .txt and .pdf files."""
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

        # Load *.txt and *.pdf files (graceful if none)
        if not os.path.isdir(DATA_PATH):
            logger.warning(
                f"DATA_PATH does not exist: {DATA_PATH} — proceeding with empty index."
            )
            file_paths: list[str] = []
        else:
            # Discover only regular files with .txt or .pdf (case-insensitive)
            file_paths: list[str] = []
            for name in os.listdir(DATA_PATH):
                full = os.path.join(DATA_PATH, name)
                if os.path.isfile(full) and name.lower().endswith((".txt", ".pdf")):
                    file_paths.append(full)

        total_files = len(file_paths)
        logger.info(f"Found {total_files} files to process (.txt, .pdf)")

        file_batch_size = 400
        chunk_size = 512
        chunk_overlap = 50
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for batch_start in tqdm(
            range(0, total_files, file_batch_size), desc="Processing file batches"
        ):
            batch_end = min(batch_start + file_batch_size, total_files)
            batch_paths = file_paths[batch_start:batch_end]
            try:
                logger.info(
                    f"Loading batch {batch_start // file_batch_size + 1} ({batch_start}-{batch_end})"
                )
                # Use PDFReader for .pdf files; default reader for .txt
                documents = SimpleDirectoryReader(
                    input_files=batch_paths,
                    encoding="utf-8",
                    file_extractor={".pdf": PyPDFReader()},
                ).load_data()
                logger.info(f"Loaded {len(documents)} documents in batch")

                batch_nodes = []
                for doc in documents:
                    # Split text into overlapping chunks for better recall
                    chunks = splitter.split_text(doc.text)
                    for chunk in chunks:
                        # Minimal, consistent metadata for traceability
                        node_metadata = {
                            "filename": os.path.basename(
                                doc.metadata.get("file_path", "")
                            ),
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
                logger.info(
                    f"Current node count in ChromaDB: {chroma_collection.count()}"
                )
                del documents, batch_nodes
            except Exception as e:
                logger.error(
                    f"Error in batch {batch_start // file_batch_size + 1}: {e}"
                )
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
            raise ValueError(
                "Error: no model loaded. Confirm a model before loading the database."
            )

        # Remove potential clamps from generate_kwargs so wrapper fields take effect
        try:
            if (
                hasattr(llm, "generate_kwargs")
                and isinstance(llm.generate_kwargs, dict)
            ):
                llm.generate_kwargs.pop("max_new_tokens", None)
                llm.generate_kwargs.pop("max_length", None)
        except Exception:
            pass

        # Ensure the LLM wrapper used by QueryEngine has the correct output budget
        try:
            n_out = int(MAX_NEW_TOKENS)
            if hasattr(llm, "max_new_tokens"):
                llm.max_new_tokens = n_out
            if hasattr(llm, "max_tokens"):
                setattr(llm, "max_tokens", n_out)
        except Exception as e:
            logger.debug(f"(UDB) Unable to set token cap on LLM wrapper: {e}")

        # Introspection to confirm no hidden clamp remains
        try:
            logger.debug(
                "(UDB) LLM before QueryEngine | type=%s, max_tokens=%s, max_new_tokens=%s, generate_kwargs=%s",
                type(llm).__name__,
                getattr(llm, "max_tokens", None),
                getattr(llm, "max_new_tokens", None),
                getattr(llm, "generate_kwargs", None),
            )
        except Exception:
            pass

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

        logger.info(
            f"ChromaDB loaded with {chroma_collection.count()} entries in {time.time() - start_time:.2f}s"
        )
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
            # Cross-version teardown: null attributes, don't delete the object here
            try:
                for name in ("_model", "model"):
                    if hasattr(llm, name):
                        setattr(llm, name, None)
                for name in ("_tokenizer", "tokenizer"):
                    if hasattr(llm, name):
                        setattr(llm, name, None)
            except Exception:
                pass
            clear_vram()
        else:
            # TO HANDLE ONLINE/OTHER LLMS
            try:
                gc.collect()
            except Exception:
                pass
        # TO RESET STATE KEYS
        state_manager.set_state("LLM", None)
        state_manager.set_state("MODEL_NAME", None)
        state_manager.set_state("MODEL_PATH", None)
        return "Model unloaded successfully."
    except Exception as e:
        logger.error(f"Unload failed: {e}")
        return f"Error unloading model: {str(e)}"


def filtered_query_engine(
    llm,
    query_str: str,
    category: str,
) -> RetrieverQueryEngine:
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
        # Remove conflicting keys so wrapper fields take effect
        try:
            if hasattr(llm, "generate_kwargs") and isinstance(llm.generate_kwargs, dict):
                llm.generate_kwargs.pop("max_new_tokens", None)
                llm.generate_kwargs.pop("max_length", None)
        except Exception:
            pass

        # Ensure LLM has the configured output budget
        try:
            n_out = int(MAX_NEW_TOKENS)
            if hasattr(llm, "max_new_tokens"):
                llm.max_new_tokens = n_out
            if hasattr(llm, "max_tokens"):
                setattr(llm, "max_tokens", n_out)
        except Exception as e:
            logger.debug(f"(UDB) Unable to set token cap on LLM wrapper: {e}")

        # Metadata filter (only effective if your nodes contain 'category' in metadata)
        meta_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="category",
                    value=category.lower(),
                    operator=FilterOperator.EQ,
                )
            ]
        )

        index = state_manager.get_state("INDEX")
        retriever = index.as_retriever(similarity_top_k=2, filters=meta_filters)

        # Configure synthesizer with provided llm and QA prompt so outputs aren't clamped
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.65)],
            response_mode="compact",
            text_qa_template=QA_PROMPT,
            streaming=False,
        )
        return query_engine
    except Exception as exc:
        logger.error(f"(UDB) filtered_query_engine failed: {exc}")
        raise



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
                    f"High usage: CPU {cpu_usage}%, RAM {memory_usage}%, GPU {gpu_usage}%â€”cleared, monitor for leaks."
                )
            logger.info(
                f"VRAM clearedâ€”CPU: {cpu_usage}%, RAM: {memory_usage}%, GPU: {gpu_usage}%"
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
            os.system("sync; echo 3 > /proc/sys/vm/drop_caches")
            logger.info(f"System RAM cleared (was {mem}% > {threshold}%)")
        except Exception as e:
            logger.warning(f"RAM clear failed: {e}. Need sudo?")
    else:
        logger.debug(f"RAM at {mem}% < {threshold}%; skipped clear")


def fetch_openai_models() -> list:
    """Fetch available OpenAI models from the API for dropdown choices, only valid chat/completion models."""
    valid_models = [
        "o1",
        "o1-2024-12-17",
        "o1-pro",
        "o1-pro-2025-03-19",
        "o1-preview",
        "o1-preview-2024-09-12",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o3-mini",
        "o3-mini-2025-01-31",
        "o3",
        "o3-2025-04-16",
        "o3-pro",
        "o3-pro-2025-06-10",
        "o4-mini",
        "o4-mini-2025-04-16",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-vision-preview",
        "gpt-4-1106-vision-preview",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-audio-preview",
        "gpt-4o-audio-preview-2024-12-17",
        "gpt-4o-audio-preview-2024-10-01",
        "gpt-4o-mini-audio-preview",
        "gpt-4o-mini-audio-preview-2024-12-17",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4.5-preview",
        "gpt-4.5-preview-2025-02-27",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-0301",
        "text-davinci-003",
        "text-davinci-002",
        "gpt-3.5-turbo-instruct",
        "text-ada-001",
        "text-babbage-001",
        "text-curie-001",
        "ada",
        "babbage",
        "curie",
        "davinci",
        "gpt-35-turbo-16k",
        "gpt-35-turbo",
        "gpt-35-turbo-0125",
        "gpt-35-turbo-1106",
        "gpt-35-turbo-0613",
        "gpt-35-turbo-16k-0613",
    ]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return valid_models
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(
            "https://api.openai.com/v1/models", headers=headers, timeout=10
        )
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
    try:
        logger.debug(f"Loading LLM for mode: {mode} with config: {config}")
        model_name = config.get("model_name") or config.get("model_path") or MODEL_PATH
        llm = create_llm(mode, model_name)
        clear_vram()
        logger.debug(f"Load success for {mode}: {model_name}")
        return f"{mode} model loaded: {model_name}", llm
    except Exception as e:
        error_msg = f"Load failed for {mode}: {str(e)}"
        logger.error(error_msg)
        return error_msg, None

def create_llm(mode: str, model_name: str) -> Optional[LLM]:
    """Create LLM instance based on mode and model name."""
    if mode not in {"Local", "HuggingFace", "OpenAI", "OpenRouter"}:
        raise ValueError("Invalid mode. Use: Local | HuggingFace | OpenAI | OpenRouter.")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty string.")

    clear_vram()
    logger.debug(f"Creating LLM for mode: {mode}, model: {model_name}")

    if mode == "Local":
        from gptqmodel import GPTQModel, QuantizeConfig
        state_manager.set_state("MODE", "Local")

        pretrained = GPTQModel.load(
            model_name,
            quantize_config=QuantizeConfig(),
            device_map=DEVICE_MAP,
            use_safetensors=True,
            trust_remote_code=True,
        )

        # Prefer GPU when available
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                pretrained.model.to("cuda:0")
                try:
                    _torch.backends.cuda.matmul.allow_tf32 = True
                    _torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not place model on GPU explicitly: {e}")

        # Cache and pad/eos sanity
        try:
            if hasattr(pretrained, "model") and hasattr(pretrained.model, "config"):
                pretrained.model.config.use_cache = True
            genconf = getattr(pretrained.model, "generation_config", None)
            if genconf is not None:
                setattr(genconf, "use_cache", True)
        except Exception as e:
            logger.debug(f"Could not enable use_cache on model: {e}")

        try:
            tok = pretrained.tokenizer
            if (
                getattr(tok, "pad_token_id", None) is None
                and getattr(tok, "eos_token_id", None) is not None
            ):
                tok.pad_token_id = int(tok.eos_token_id)
        except Exception as e:
            logger.debug(f"Could not set pad_token_id on tokenizer: {e}")

        # Dynamically derive EOS stop list (safe across models)
        eos_ids = None
        try:
            eos_tok = getattr(pretrained.tokenizer, "eos_token_id", None)
            if isinstance(eos_tok, int):
                eos_ids = [int(eos_tok)]
            elif isinstance(eos_tok, (list, tuple)) and eos_tok:
                eos_ids = [int(x) for x in eos_tok if isinstance(x, int)]
        except Exception as e:
            logger.debug(f"Could not derive eos_token_id: {e}")

        llm = HuggingFaceLLM(
            model=pretrained.model,
            tokenizer=pretrained.tokenizer,
            context_window=int(CONTEXT_WINDOW),
            max_new_tokens=int(MAX_NEW_TOKENS),
            system_prompt="",
            generate_kwargs={
                "temperature": float(TEMPERATURE),
                "top_p": float(TOP_P),
                "top_k": int(TOP_K),
                "repetition_penalty": float(REPETITION_PENALTY),
                "no_repeat_ngram_size": int(NO_REPEAT_NGRAM_SIZE),
                "do_sample": True,
                "use_cache": True,
                "pad_token_id": int(
                    getattr(
                        pretrained.tokenizer,
                        "pad_token_id",
                        getattr(pretrained.tokenizer, "eos_token_id", 0),
                    )
                ),
            },
            device_map=DEVICE_MAP,
            is_chat_model=False,
            stopping_ids=eos_ids,  # ensure early stop on EOS
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("HF Local LLM ready (EOS stopping enabled).")
        return llm

    if mode == "HuggingFace":
        state_manager.set_state("MODE", "Online")
        llm = HuggingFaceInferenceAPI(
            model_name=model_name,
            api_key=__import__("os").getenv("HF_TOKEN"),
            provider="auto",
            is_chat_model=False,
            num_output=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
        )
        token = __import__("os").getenv("HF_TOKEN")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            llm.__pydantic_private__["_tokenizer"] = tokenizer
            logger.debug("Tokenizer attached to HuggingFaceInferenceAPI wrapper.")
        except Exception as e:
            logger.warning(f"Tokenizer load failed for {model_name}: {e}")
        state_manager.set_state("USE_CHAT", False)
        logger.info("HF Inference API LLM ready.")
        return llm

    if mode == "OpenAI":
        state_manager.set_state("MODE", "Online")
        llm = LlamaOpenAI(
            model=model_name,
            api_key=__import__("os").getenv("OPENAI_API_KEY"),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            max_tokens=int(MAX_NEW_TOKENS),
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("OpenAI LLM ready.")
        return llm

    if mode == "OpenRouter":
        state_manager.set_state("MODE", "Online")
        llm = OpenRouter(
            model=model_name,
            api_key=__import__("os").getenv("OPENROUTER_API_KEY"),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            max_tokens=int(MAX_NEW_TOKENS),
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("OpenRouter LLM ready.")
        return llm

    raise ValueError(f"Unhandled mode: {mode}")


def create_llm(mode: str, model_name: str) -> Optional[LLM]:
    """Create LLM instance based on mode and model name."""
    # Validate inputs
    if mode not in {"Local", "HuggingFace", "OpenAI", "OpenRouter"}:
        raise ValueError(
            "Invalid mode. Use: Local | HuggingFace | OpenAI | OpenRouter."
        )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Model name must be a non-empty string.")

    # Prep
    clear_vram()
    logger.debug(f"Creating LLM for mode: {mode}, model: {model_name}")

    if mode == "Local":
        from gptqmodel import GPTQModel, QuantizeConfig  # localize import for clarity
        state_manager.set_state("MODE", "Local")

        pretrained = GPTQModel.load(
            model_name,
            quantize_config=QuantizeConfig(),
            device_map=DEVICE_MAP,
            use_safetensors=True,
            trust_remote_code=True,
        )

        try:
            import torch as _torch

            if _torch.cuda.is_available():
                pretrained.model.to("cuda:0")
                try:
                    _torch.backends.cuda.matmul.allow_tf32 = True
                    _torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Could not place model on GPU explicitly: {e}")

        try:
            if hasattr(pretrained, "model") and hasattr(pretrained.model, "config"):
                pretrained.model.config.use_cache = True
            genconf = getattr(pretrained.model, "generation_config", None)
            if genconf is not None:
                setattr(genconf, "use_cache", True)
        except Exception as e:
            logger.debug(f"Could not enable use_cache on model: {e}")
        try:
            tok = pretrained.tokenizer
            if (
                getattr(tok, "pad_token_id", None) is None
                and getattr(tok, "eos_token_id", None) is not None
            ):
                tok.pad_token_id = int(tok.eos_token_id)
        except Exception as e:
            logger.debug(f"Could not set pad_token_id on tokenizer: {e}")

        llm = HuggingFaceLLM(
            model=pretrained.model,
            tokenizer=pretrained.tokenizer,
            context_window=int(CONTEXT_WINDOW),
            max_new_tokens=int(MAX_NEW_TOKENS),
            system_prompt="",
            generate_kwargs={
                "temperature": float(TEMPERATURE),
                "top_p": float(TOP_P),
                "top_k": int(TOP_K),
                "repetition_penalty": float(REPETITION_PENALTY),
                "no_repeat_ngram_size": int(NO_REPEAT_NGRAM_SIZE),
                "do_sample": True,
                "use_cache": True,
                "pad_token_id": int(
                    getattr(
                        pretrained.tokenizer,
                        "pad_token_id",
                        getattr(pretrained.tokenizer, "eos_token_id", 0),
                    )
                ),
            },
            device_map=DEVICE_MAP,
            is_chat_model=False,
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("HF Local LLM ready (GPU placement enforced; KV cache enabled).")
        return llm

    if mode == "HuggingFace":
        state_manager.set_state("MODE", "Online")
        llm = HuggingFaceInferenceAPI(
            model_name=model_name,
            api_key=__import__("os").getenv("HF_TOKEN"),
            provider="auto",
            is_chat_model=False,
            num_output=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
        )
        token = __import__("os").getenv("HF_TOKEN")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            llm.__pydantic_private__["_tokenizer"] = tokenizer
            logger.debug(
                f"Tokenizer loaded and attached to __pydantic_private__ for {model_name}."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer for {model_name}: {str(e)}. Token counting may be approximate."
            )
        state_manager.set_state("USE_CHAT", False)
        logger.info(
            "HF Inference API LLM ready (single cap via constructor; chat disabled)."
        )
        return llm

    if mode == "OpenAI":
        state_manager.set_state("MODE", "Online")
        llm = LlamaOpenAI(
            model=model_name,
            api_key=__import__("os").getenv("OPENAI_API_KEY"),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            max_tokens=int(MAX_NEW_TOKENS),
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("OpenAI LLM ready (cap via max_tokens in constructor).")
        return llm

    if mode == "OpenRouter":
        state_manager.set_state("MODE", "Online")
        llm = OpenRouter(
            model=model_name,
            api_key=__import__("os").getenv("OPENROUTER_API_KEY"),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            max_tokens=int(MAX_NEW_TOKENS),
        )
        state_manager.set_state("USE_CHAT", True)
        logger.info("OpenRouter LLM ready (cap via max_tokens in constructor).")
        return llm

    # Unreachable due to validation
    raise ValueError(f"Unhandled mode: {mode}")

def stream_generate(
    query: str,
    system_prompt: str,
    llm: Any,
    *,
    max_new_tokens: Optional[int] = None,
    think: bool = False,
    depth: int = 0,
    **kwargs: Any,
) -> str:
    """Simple streaming generation with strict token limit from config."""
    user_text = (query or "").strip()
    sys_text = (system_prompt or "").strip()
    if not sys_text:
        try:
            sys_text = system_prefix().strip()
        except Exception:
            sys_text = ""

    if not user_text and not sys_text:
        return "EMPTY PROMPT."

    model_name = getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""
    control_tag = ""
    if model_name and any(x in str(model_name) for x in ("CyberSic", "Qwen3")):
        control_tag = "/think" if think else "/no_think"

    if control_tag:
        prompt = f"{sys_text}\n{control_tag}\n{user_text}" if sys_text else f"{control_tag}\n{user_text}"
    else:
        prompt = f"{sys_text}\n{user_text}" if sys_text else user_text

    # arg -> config.toml (via config.py) -> config.py default
    target_tokens: int = int(max_new_tokens) if max_new_tokens is not None else int(MAX_NEW_TOKENS)

    # Remove conflicting keys from kwargs
    call_kwargs: Dict[str, Any] = dict(kwargs) if kwargs else {}
    for k in ("max_tokens", "max_new_tokens", "max_output_tokens", "max_length"):
        call_kwargs.pop(k, None)

    # Sync wrapper fields so nothing clamps to 256
    prev_max = getattr(llm, "max_new_tokens", None)
    try:
        if hasattr(llm, "max_new_tokens"):
            llm.max_new_tokens = target_tokens
        if hasattr(llm, "max_tokens"):
            setattr(llm, "max_tokens", target_tokens)
        if hasattr(llm, "generate_kwargs") and isinstance(llm.generate_kwargs, dict):
            llm.generate_kwargs["max_new_tokens"] = target_tokens
            llm.generate_kwargs.pop("max_length", None)
    except Exception:
        pass

    # Choose per-call key by provider
    name = type(llm).__name__.lower()
    use_max_new = any(s in name for s in ("huggingface", "transformer", "hf"))
    budget_key = "max_new_tokens" if use_max_new else "max_tokens"

    try:
        # Streaming first
        stream_fn = getattr(llm, "stream_complete", None)
        if callable(stream_fn):
            try:
                try:
                    stream = stream_fn(prompt, **{budget_key: target_tokens}, **call_kwargs)
                except TypeError:
                    stream = stream_fn(prompt, **call_kwargs)

                parts: list[str] = []
                for chunk in stream:
                    delta = getattr(chunk, "delta", None)
                    if delta:
                        parts.append(str(delta))
                text = "".join(parts).strip()

                if not text:
                    aggregated = getattr(stream, "text", "")
                    text = str(aggregated).strip() if aggregated else ""

                return text or "EMPTY RESPONSE FROM MODEL."
            except Exception:
                logger.debug("stream_complete failed; using complete")

        # Fallback non-stream
        complete_fn = getattr(llm, "complete", None)
        if callable(complete_fn):
            try:
                try:
                    resp = complete_fn(prompt, **{budget_key: target_tokens}, **call_kwargs)
                except TypeError:
                    resp = complete_fn(prompt, **call_kwargs)
                text = getattr(resp, "text", None)
                if text is None:
                    text = str(resp)
                return (text or "").strip() or "EMPTY RESPONSE FROM MODEL."
            except Exception as e:
                logger.error(f"GENERATION FAILURE — {e}")
                return f"GENERATION FAILURE — {e}"

        logger.error("LLM BACKEND UNSUPPORTED FOR GENERATION.")
        return "LLM BACKEND UNSUPPORTED FOR GENERATION."
    except Exception as e:
        logger.error(f"GENERATION FAILURE — {e}")
        return f"GENERATION FAILURE — {e}"
    finally:
        clear_vram()
        if max_new_tokens is not None and prev_max is not None and hasattr(llm, "max_new_tokens"):
            try:
                llm.max_new_tokens = prev_max
            except Exception:
                pass

def generate(
    query: str,
    system_prompt: str,
    llm: Any,
    *,
    max_new_tokens: Optional[int] = None,
    think: bool = False,
    depth: int = 0,
    **kwargs: Any,
) -> str:
    """Generate a response using the LLM with OOM handling and provider-correct token caps."""
    if depth > 3:
        return "Error: Max OOM retry depth exceeded—clear VRAM manually."

    system_prompt = (system_prompt or "").strip()

    # Determine control tag only for specific models
    model_name = getattr(llm, "model_name", "") or getattr(llm, "model", "") or ""
    control_tag = ""
    if model_name and any(
        x.lower() in str(model_name).lower() for x in ("cybersic", "qwen3")
    ):
        control_tag = "/think" if think else "/no_think"

    # Optional dynamic LLM selection via kwargs (engine+model)
    dyn_model = kwargs.get("model")
    dyn_engine = kwargs.get("engine")
    if dyn_model and dyn_engine:
        try:
            llm = create_llm(dyn_engine, dyn_model)
        except Exception as e:
            logger.warning(f"Dynamic LLM load failed for '{dyn_engine}': {e}")
            return f"Error loading specified model: {str(e)}"

    # Token budget
    target = (
        int(max_new_tokens)
        if isinstance(max_new_tokens, int) and max_new_tokens > 0
        else int(MAX_NEW_TOKENS)
    )

    # Remove potential clamps from generate_kwargs so wrapper fields take effect and avoid duplicates
    try:
        if hasattr(llm, "generate_kwargs") and isinstance(llm.generate_kwargs, dict):
            llm.generate_kwargs.pop("max_new_tokens", None)
            llm.generate_kwargs.pop("max_output_tokens", None)
            llm.generate_kwargs.pop("max_length", None)
    except Exception:
        pass

    # Best-effort: set wrapper caps; restore in finally
    prev_max_tokens = getattr(llm, "max_tokens", None) if hasattr(llm, "max_tokens") else None
    prev_max_new = getattr(llm, "max_new_tokens", None) if hasattr(llm, "max_new_tokens") else None
    try:
        if hasattr(llm, "max_tokens"):
            setattr(llm, "max_tokens", target)
        if hasattr(llm, "max_new_tokens"):
            setattr(llm, "max_new_tokens", target)

        # Provider-aware key for per-call budget (used where applicable)
        lower_name = type(llm).__name__.lower()
        if "huggingfaceinferenceapi" in lower_name:
            budget_key = "max_tokens"
        elif any(s in lower_name for s in ("huggingface", "transformer", "hf")):
            budget_key = "max_new_tokens"
        else:
            budget_key = "max_tokens"

        # Build prompt and chat messages
        user_query = query or ""
        prompt_parts = [p for p in (system_prompt, control_tag, user_query) if p]
        full_prompt = "\n".join(prompt_parts) if prompt_parts else user_query

        chat: list[ChatMessage] = []
        if system_prompt:
            chat.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        if control_tag:
            chat.append(ChatMessage(role=MessageRole.SYSTEM, content=control_tag))
        chat.append(ChatMessage(role=MessageRole.USER, content=user_query))

        # Choose chat vs complete based on state
        use_chat = bool(state_manager.get_state("USE_CHAT", True))

        # Debug introspection of LLM caps to detect hidden clamps
        try:
            logger.debug(
                "GEN: llm=%s max_tokens=%s max_new_tokens=%s gen_kwargs=%s",
                type(llm).__name__,
                getattr(llm, "max_tokens", None),
                getattr(llm, "max_new_tokens", None),
                getattr(llm, "generate_kwargs", None)
                if hasattr(llm, "generate_kwargs")
                else None,
            )
        except Exception:
            pass

        start_time = time.time()
        if use_chat and callable(getattr(llm, "chat", None)):
            resp = llm.chat(chat)
            text_out = ""
            if hasattr(resp, "message") and getattr(resp, "message") is not None:
                msg = getattr(resp, "message")
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    text_out = msg.content
            elif hasattr(resp, "text") and isinstance(resp.text, str):
                text_out = resp.text
            else:
                text_out = str(resp)

            if not (text_out or "").strip() and callable(getattr(llm, "complete", None)):
                logger.warning("Empty chat response—falling back to complete.")
                try:
                    r2 = llm.complete(full_prompt, **{budget_key: target})
                except TypeError:
                    r2 = llm.complete(full_prompt)
                text_out = r2.text if hasattr(r2, "text") else str(r2)
        elif callable(getattr(llm, "complete", None)):
            try:
                resp = llm.complete(full_prompt, **{budget_key: target})
            except TypeError:
                resp = llm.complete(full_prompt)
            text_out = resp.text if hasattr(resp, "text") else str(resp)
        else:
            return "LLM BACKEND UNSUPPORTED FOR GENERATION."

        text_out = (text_out or "").strip()
        if not text_out:
            logger.warning(
                "Empty response from model—check query, provider limits, or model compatibility."
            )
            return "Empty response from model."

        # Metrics (lightweight)
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            total_tokens = len(enc.encode(text_out))
        except Exception as e:
            logger.warning(f"Tiktoken failed: {e} - fallback to word count.")
            total_tokens = len(text_out.split())

        duration = time.time() - start_time
        tps = total_tokens / duration if duration > 0 else 0.0

        run_count = (state_manager.get_state("run_count") or 0) + 1
        total_tokens_all = (state_manager.get_state("total_tokens_all") or 0) + total_tokens
        total_duration_all = (state_manager.get_state("total_duration_all") or 0.0) + duration
        state_manager.set_state("run_count", run_count)
        state_manager.set_state("total_tokens_all", total_tokens_all)
        state_manager.set_state("total_duration_all", total_duration_all)
        avg_tps = total_tokens_all / total_duration_all if total_duration_all > 0 else 0.0

        logger.info(
            f"Inference Time: {duration:.2f}s | Tokens: {total_tokens} | TPS: {tps:.2f} | Avg TPS: {avg_tps:.2f}"
        )
        return text_out

    except torch.cuda.OutOfMemoryError as e:
        logger.warning(
            f"Error: OOM during generation - {e}. Reduce max_tokens or clear VRAM."
        )
        clear_vram()
        # Reduce target and retry
        new_target = max(32, (target // 2))
        try:
            if hasattr(llm, "max_tokens"):
                setattr(llm, "max_tokens", new_target)
            if hasattr(llm, "max_new_tokens"):
                setattr(llm, "max_new_tokens", new_target)
        except Exception:
            pass
        return generate(
            query,
            system_prompt,
            llm,
            max_new_tokens=new_target,
            think=think,
            depth=depth + 1,
            **kwargs,
        )
    except Exception as e:
        err = f"Error: Unable to generate response: {e}"
        logger.error(err)
        return err
    finally:
        clear_vram()
        try:
            if prev_max_new is not None and hasattr(llm, "max_new_tokens"):
                setattr(llm, "max_new_tokens", prev_max_new)
        except Exception:
            pass
        try:
            if prev_max_tokens is not None and hasattr(llm, "max_tokens"):
                setattr(llm, "max_tokens", prev_max_tokens)
        except Exception:
            pass


__all__ = [
    "stream_generate",
    "update_database",
    "clear_vram",
    "save_database",
    "load_database",
    "get_system_info",
    "append_to_chatbot",
    "filtered_query_engine",
    "clear_ram",
    "fetch_openai_models",
    "create_llm",
    "load_llm",
    "unload_model",
]

class _Buf:
    """Small time/semantic buffered emitter to reduce UI churn.

    - Flushes on 1s timer, sentence end (., !, ?), paragraph break (\n\n),
      or when max_buf_len is exceeded.
    - Stateless to callers other than push()/flush().
    """

    def __init__(
        self,
        *,
        interval: float = 1.0,
        max_buf_len: int = 1200,
    ) -> None:
        self._buf: List[str] = []
        self._t0: float = time.monotonic()
        self._interval = float(max(0.1, interval))
        self._max = int(max(64, max_buf_len))

    def _due(self) -> bool:
        return (time.monotonic() - self._t0) >= self._interval

    def _semantic_boundary(self, s: str) -> bool:
        if not s:
            return False
        # Sentence end or paragraph break
        if s.endswith((".", "!", "?")):
            return True
        if "\n\n" in s:
            return True
        return False

    def push(self, piece: str) -> Optional[str]:
        piece = str(piece or "")
        if not piece:
            return None
        self._buf.append(piece)
        cur = "".join(self._buf)
        if len(cur) >= self._max or self._semantic_boundary(cur) or self._due():
            out = cur
            self._buf.clear()
            self._t0 = time.monotonic()
            return out
        return None

    def flush(self) -> Optional[str]:
        if not self._buf:
            return None
        out = "".join(self._buf)
        self._buf.clear()
        self._t0 = time.monotonic()
        return out


def iter_buffered(
    stream: Iterable[str],
    *,
    interval: float = 1.0,
    max_buf_len: int = 1200,
) -> Generator[str, None, None]:
    """Yield buffered chunks from a token/text iterable with 1s timer and
    sentence/paragraph boundary flushes.
    """
    buf = _Buf(interval=interval, max_buf_len=max_buf_len)
    for tok in stream:
        flushed = buf.push(tok)
        if flushed:
            yield flushed
    tail = buf.flush()
    if tail:
        yield tail
