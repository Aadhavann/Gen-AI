import os
import pandas as pd
import numpy as np
from nbformat import reads, NO_CONVERT
from tqdm import tqdm
from datasets import Dataset
from typing import Dict, List, Tuple, Generator, Optional
from huggingface_hub import create_repo, upload_folder
import tempfile
import subprocess
import multiprocessing as mp
from functools import partial
import pathlib
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

MIRROR_DIRECTORY = "hf_public_repos"
DATASET_ID = "hf-codegen"
SERIALIZE_IN_CHUNKS = 10000
FEATHER_FORMAT = "ftr"
MAX_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one CPU free for system operations

# Block the following formats as a set for faster lookups
IMAGE = {"png", "jpg", "jpeg", "gif"}
VIDEO = {"mp4", "jfif"}
DOC = {"key", "PDF", "pdf", "docx", "xlsx", "pptx"}
AUDIO = {"flac", "ogg", "mid", "webm", "wav", "mp3"}
ARCHIVE = {"jar", "aar", "gz", "zip", "bz2"}
MODEL = {"onnx", "pickle", "model", "neuron"}
OTHERS = {
    "npy", "index", "inv", "DS_Store", "rdb", "pack", 
    "idx", "glb", "gltf", "len", "otf", "unitypackage", 
    "ttf", "xz", "pcm", "opus"
}

# Combine all blocked formats into a single set for O(1) lookups
BLOCKED_EXTENSIONS = IMAGE | VIDEO | DOC | AUDIO | ARCHIVE | MODEL | OTHERS
BLOCKED_PATTERNS = {".git", "__pycache__", "xcodeproj"}


def upload_to_hub(file_format: str, repo_id: str) -> None:
    """Moves all the files matching `file_format` to a folder and
    uploads the folder to the Hugging Face Hub."""
    try:
        repo_id = create_repo(repo_id=repo_id, exist_ok=True, repo_type="dataset").repo_id
        logger.info(f"Created/verified repository: {repo_id}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Use pathlib for more robust path handling
            tmp_path = pathlib.Path(tmpdirname)
            os.makedirs(tmp_path, exist_ok=True)
            
            # Use subprocess with shell=False for better security
            command = ["mv", f"*.{file_format}", str(tmp_path)]
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            
            if result.returncode != 0 and "No such file or directory" not in result.stderr:
                logger.warning(f"Issue during file move: {result.stderr}")
                
            upload_folder(repo_id=repo_id, folder_path=str(tmp_path), repo_type="dataset")
            logger.info(f"Successfully uploaded folder to {repo_id}")
    except Exception as e:
        logger.error(f"Error in upload_to_hub: {str(e)}")
        raise


def filter_code_cell(cell) -> bool:
    """Filters a code cell w.r.t shell commands, etc."""
    source = cell.get("source", "")
    # Fast path for empty cells
    if not source:
        return False
    
    return not (source.startswith("!") or "%%capture" in source)


def process_notebook(content: str) -> str:
    """Extract and concatenate code from notebook cells."""
    try:
        notebook = reads(content, NO_CONVERT)
        
        # Use list comprehension with filtering for better performance
        code_cells = [
            c["source"] for c in notebook.get("cells", [])
            if c.get("cell_type") == "code" and filter_code_cell(c)
        ]
        
        # Join is more efficient than string concatenation in a loop
        return "".join(code_cells)
    except Exception as e:
        logger.debug(f"Error processing notebook: {str(e)}")
        return ""


def process_file(file_tuple: Tuple[str, str]) -> Optional[Dict[str, str]]:
    """Processes a single file and returns its content in a structured format."""
    directory_name, file_path = file_tuple
    
    try:
        # Use pathlib for more reliable path operations
        path = pathlib.Path(file_path)
        
        # Skip processing if the file is too large (e.g., > 10MB)
        if path.stat().st_size > 10_000_000:
            return None
            
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            
            # Process notebooks
            if file_path.endswith("ipynb"):
                content = process_notebook(content)
                
            # Skip if content is empty after processing
            if not content:
                return None
                
            return {
                "repo_id": directory_name,
                "file_path": file_path,
                "content": content,
            }
    except (UnicodeDecodeError, PermissionError, FileNotFoundError):
        # Common file reading errors - skip silently
        return None
    except Exception as e:
        # Log other unexpected errors but continue processing
        logger.debug(f"Error processing {file_path}: {str(e)}")
        return None


def is_valid_file(file_path: str) -> bool:
    """Quickly check if a file should be processed based on its path."""
    path = pathlib.Path(file_path)
    
    # Skip files with blocked extensions
    if path.suffix.lstrip('.') in BLOCKED_EXTENSIONS:
        return False
        
    # Skip files in blocked directories
    if any(pattern in file_path for pattern in BLOCKED_PATTERNS):
        return False
        
    return True


def find_files(directory: str) -> Generator[Tuple[str, str], None, None]:
    """Generate valid file paths lazily to avoid storing everything in memory."""
    for root, _, files in os.walk(directory):
        repo_dir = os.path.dirname(root)
        for file in files:
            file_path = os.path.join(root, file)
            if is_valid_file(file_path):
                yield (repo_dir, file_path)


def process_files_batch(file_batch: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """Process a batch of files and return valid results."""
    results = []
    for file_tuple in file_batch:
        result = process_file(file_tuple)
        if result:
            results.append(result)
    return results


def read_repository_files(directory: str) -> None:
    """Reads files from repositories and saves them in chunks."""
    try:
        # Get all valid file paths lazily
        file_generator = find_files(directory)
        
        # Count files first to set up progress bar
        logger.info("Counting files for processing...")
        file_paths = list(file_generator)
        total_files = len(file_paths)
        logger.info(f"Total file paths: {total_files}")
        
        # Process in batches for better memory management
        batch_size = min(1000, total_files // (MAX_WORKERS * 2) + 1)
        chunk_flag = 0
        total_processed = 0
        
        # Set up multiprocessing pool
        with mp.Pool(processes=MAX_WORKERS) as pool:
            for i in range(0, total_files, batch_size):
                batch = file_paths[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1} ({len(batch)} files)")
                
                # Process batch in parallel
                results = pool.map(process_file, batch)
                
                # Filter out None results and create DataFrame
                valid_results = [r for r in results if r is not None]
                total_processed += len(valid_results)
                
                if valid_results:
                    # Create DataFrame from batch results
                    df_batch = pd.DataFrame(valid_results)
                    
                    # Save to feather if we have enough records or it's the last batch
                    if len(df_batch) >= SERIALIZE_IN_CHUNKS or i + batch_size >= total_files:
                        df_path = f"df_chunk_{chunk_flag}_{len(df_batch)}.{FEATHER_FORMAT}"
                        logger.info(f"Serializing dataframe to {df_path}...")
                        df_batch.reset_index(drop=True).to_feather(df_path)
                        chunk_flag += 1
        
        logger.info(f"Processed {total_processed} files successfully")
    
    except Exception as e:
        logger.error(f"Error in read_repository_files: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info(f"Starting dataset generation from {MIRROR_DIRECTORY}")
    logger.info(f"Using {MAX_WORKERS} parallel workers")
    
    # Process files and create chunked feather files
    read_repository_files(MIRROR_DIRECTORY)
    
    # Upload files to Hub
    logger.info("Uploading chunked files to the Hub...")
    upload_to_hub(file_format=FEATHER_FORMAT, repo_id=DATASET_ID)
    logger.info(f"{FEATHER_FORMAT} files uploaded to the Hub at {DATASET_ID}")
    
    if not SERIALIZE_IN_CHUNKS:
        logger.info("SERIALIZE_IN_CHUNKS is disabled - creating complete dataset...")
        # Read all feather files and combine
        feather_files = [f for f in os.listdir('.') if f.endswith(f'.{FEATHER_FORMAT}')]
        dfs = [pd.read_feather(f) for f in feather_files]
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            dataset = Dataset.from_pandas(combined_df)
            dataset.push_to_hub(DATASET_ID)
            logger.info(f"Complete dataset pushed to {DATASET_ID}")
        else:
            logger.warning("No data files found to create dataset")