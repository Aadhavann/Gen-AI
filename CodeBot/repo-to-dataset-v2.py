import os
import pandas as pd
import json
import ast
import re
import gitpython as git
from nbformat import reads, NO_CONVERT
from tqdm import tqdm
from datasets import Dataset
from typing import Dict, List, Tuple, Optional, Union
from huggingface_hub import HfApi, create_repo
import tempfile
import subprocess
import logging
from pathlib import Path
import random
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MIRROR_DIRECTORY = "repo_mirror"
DATASET_ID = "repo-finetuning-dataset"
SERIALIZE_IN_CHUNKS = 10000
FEATHER_FORMAT = "ftr"

# Block the following formats
IMAGE = ["png", "jpg", "jpeg", "gif"]
VIDEO = ["mp4", "jfif"]
DOC = ["key", "PDF", "pdf", "docx", "xlsx", "pptx"]
AUDIO = ["flac", "ogg", "mid", "webm", "wav", "mp3"]
ARCHIVE = ["jar", "aar", "gz", "zip", "bz2"]
MODEL = ["onnx", "pickle", "model", "neuron"]
OTHERS = [
    "npy", "index", "inv", "index", "DS_Store", "rdb", "pack", 
    "idx", "glb", "gltf", "len", "otf", "unitypackage", "ttf", 
    "xz", "pcm", "opus"
]
EXCLUDED_FORMATS = tuple(IMAGE + VIDEO + DOC + AUDIO + ARCHIVE + MODEL + OTHERS)

# Supported languages for code analysis
SUPPORTED_LANGUAGES = {
    'py': 'python',
    'js': 'javascript',
    'ts': 'typescript',
    'java': 'java',
    'c': 'c',
    'cpp': 'cpp',
    'cs': 'csharp',
    'go': 'go',
    'rb': 'ruby',
    'php': 'php',
    'rs': 'rust',
    'scala': 'scala',
    'swift': 'swift',
    'kt': 'kotlin',
}

# Language-specific comment patterns
COMMENT_PATTERNS = {
    'python': {
        'single_line': r'#.*',
        'multi_line': r'""".*?"""',
        'docstring': r'""".*?"""',
    },
    'javascript': {
        'single_line': r'//.*',
        'multi_line': r'/\*.*?\*/',
        'docstring': r'/\*\*.*?\*/',
    },
    'typescript': {
        'single_line': r'//.*',
        'multi_line': r'/\*.*?\*/',
        'docstring': r'/\*\*.*?\*/',
    },
    'java': {
        'single_line': r'//.*',
        'multi_line': r'/\*.*?\*/',
        'docstring': r'/\*\*.*?\*/',
    },
    # Add patterns for other languages as needed
}


def clone_repository(repo_url: str, target_dir: str = MIRROR_DIRECTORY) -> str:
    logger.info(f"Cloning repository: {repo_url}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Extract repo name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(target_dir, repo_name)
    
    if os.path.exists(repo_path):
        logger.info(f"Repository already exists at {repo_path}")
        return repo_path
    
    try:
        git.Repo.clone_from(repo_url, repo_path)
        logger.info(f"Repository cloned successfully to {repo_path}")
        return repo_path
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        raise


def extract_metadata(repo_path: str) -> Dict:
    try:
        repo = git.Repo(repo_path)
        return {
            'name': os.path.basename(repo_path),
            'description': '',  # Could be fetched from GitHub API if available
            'default_branch': repo.active_branch.name,
            'last_commit': str(repo.head.commit.hexsha),
            'commit_date': str(repo.head.commit.committed_datetime),
            'contributors': [c.author.name for c in list(repo.iter_commits(max_count=10))],
        }
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        return {'name': os.path.basename(repo_path)}


def parse_directory_structure(repo_path: str) -> Dict:
    structure = {'files': [], 'directories': {}}
    
    for root, dirs, files in os.walk(repo_path):
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            current = structure
        else:
            current = structure
            for part in rel_path.split(os.sep):
                if part == '.git' or part == '__pycache__':
                    current = None
                    break
                if current is not None:
                    if part not in current['directories']:
                        current['directories'][part] = {'files': [], 'directories': {}}
                    current = current['directories'][part]
        
        if current is not None:
            current['files'].extend([f for f in files if not f.endswith(EXCLUDED_FORMATS)])
    
    return structure


def filter_code_files(structure: Dict, repo_path: str) -> List[str]:
    code_files = []
    
    def traverse(struct, path):
        for file in struct['files']:
            file_ext = file.split('.')[-1] if '.' in file else ''
            if file_ext in SUPPORTED_LANGUAGES:
                file_path = os.path.join(path, file)
                code_files.append(file_path)
        
        for dir_name, dir_struct in struct['directories'].items():
            traverse(dir_struct, os.path.join(path, dir_name))
    
    traverse(structure, repo_path)
    return code_files


def filter_documentation_files(structure: Dict, repo_path: str) -> List[str]:
    doc_files = []
    
    def traverse(struct, path):
        for file in struct['files']:
            lower_file = file.lower()
            if (lower_file.endswith('.md') or 
                lower_file == 'readme' or 
                'doc' in lower_file or 
                'tutorial' in lower_file):
                file_path = os.path.join(path, file)
                doc_files.append(file_path)
        
        for dir_name, dir_struct in struct['directories'].items():
            traverse(dir_struct, os.path.join(path, dir_name))
    
    traverse(structure, repo_path)
    return doc_files


def filter_code_cell(cell) -> bool:
    only_shell = cell["source"].startswith("!")
    only_magic = "%%capture" in cell["source"]
    if only_shell or only_magic:
        return False
    else:
        return True


def process_file(file_path: str) -> Dict[str, str]:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            
            if file_path.endswith("ipynb"):
                # Process Jupyter notebooks
                code_cell_str = ""
                notebook = reads(content, NO_CONVERT)

                code_cells = [
                    c for c in notebook["cells"]
                    if c["cell_type"] == "code" and filter_code_cell(c)
                ]

                for cell in code_cells:
                    code_cell_str += cell["source"] + "\n\n"
                content = code_cell_str
                
            file_ext = file_path.split('.')[-1] if '.' in file_path else ''
            language = SUPPORTED_LANGUAGES.get(file_ext, 'unknown')
            
            return {
                "file_path": file_path,
                "language": language,
                "content": content,
            }
    except Exception as e:
        logger.warning(f"Failed to process file {file_path}: {e}")
        return {
            "file_path": file_path,
            "language": "unknown",
            "content": "",
        }


def extract_comments_and_docstrings(file_content: str, language: str) -> Tuple[List[str], List[str]]:
    if language not in COMMENT_PATTERNS:
        return [], []
    
    patterns = COMMENT_PATTERNS[language]
    
    comments = []
    for match in re.finditer(patterns['single_line'], file_content, re.DOTALL):
        comment = match.group(0).strip()
        if comment and len(comment) > 3:  # Filter out very short comments
            comments.append(comment)
    
    docstrings = []
    for match in re.finditer(patterns['docstring'], file_content, re.DOTALL):
        docstring = match.group(0).strip()
        if docstring and len(docstring) > 10:  # Filter out very short docstrings
            docstrings.append(docstring)
    
    return comments, docstrings


def parse_python_code(file_content: str) -> List[Dict]:
    symbols = []
    
    try:
        tree = ast.parse(file_content)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get function docstring if available
                docstring = ast.get_docstring(node) or ""
                
                # Get function code
                function_lines = file_content.splitlines()[node.lineno-1:node.end_lineno]
                function_code = '\n'.join(function_lines)
                
                # Get function parameters
                params = []
                for arg in node.args.args:
                    params.append(arg.arg)
                
                symbols.append({
                    'type': 'function',
                    'name': node.name,
                    'docstring': docstring,
                    'code': function_code,
                    'parameters': params,
                    'lineno': node.lineno,
                })
                
            elif isinstance(node, ast.ClassDef):
                # Get class docstring if available
                docstring = ast.get_docstring(node) or ""
                
                # Get class code
                class_lines = file_content.splitlines()[node.lineno-1:node.end_lineno]
                class_code = '\n'.join(class_lines)
                
                # Get class methods
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_docstring = ast.get_docstring(child) or ""
                        methods.append({
                            'name': child.name,
                            'docstring': method_docstring,
                        })
                
                symbols.append({
                    'type': 'class',
                    'name': node.name,
                    'docstring': docstring,
                    'code': class_code,
                    'methods': methods,
                    'lineno': node.lineno,
                })
    
    except SyntaxError:
        logger.warning(f"Failed to parse Python code, syntax error")
    except Exception as e:
        logger.warning(f"Failed to parse Python code: {e}")
    
    return symbols


def analyze_code_file(file_data: Dict) -> Dict:
    result = file_data.copy()
    content = file_data['content']
    language = file_data['language']
    
    # Extract comments and docstrings
    comments, docstrings = extract_comments_and_docstrings(content, language)
    result['comments'] = comments
    result['docstrings'] = docstrings
    
    # For Python files, extract functions and classes
    if language == 'python':
        result['symbols'] = parse_python_code(content)
    else:
        result['symbols'] = []  # Could add parsers for other languages
    
    return result


def generate_code_completion_examples(analyzed_files: List[Dict]) -> List[Dict]:
    completion_examples = []
    
    for file_data in analyzed_files:
        content = file_data['content']
        language = file_data['language']
        
        if not content or language == 'unknown':
            continue
        
        symbols = file_data.get('symbols', [])
        
        # Generate function completion examples
        for symbol in symbols:
            if symbol['type'] == 'function':
                code = symbol['code']
                
                # If the function is substantial enough
                if len(code.splitlines()) >= 5:
                    # Get the function signature and half of the body
                    lines = code.splitlines()
                    split_idx = min(len(lines) // 2 + 1, len(lines) - 1)
                    
                    prompt = '\n'.join(lines[:split_idx])
                    completion = '\n'.join(lines[split_idx:])
                    
                    completion_examples.append({
                        'type': 'code_completion',
                        'language': language,
                        'instruction': f"Complete the following {language} function:\n\n{prompt}",
                        'response': completion,
                        'metadata': {
                            'file_path': file_data['file_path'],
                            'function_name': symbol['name'],
                        }
                    })
        
        # Generate line-by-line completion examples for files with substantial content
        lines = content.splitlines()
        if len(lines) >= 10:
            for i in range(5, len(lines) - 5, 10):  # Sample every 10 lines
                prompt_lines = lines[:i]
                completion_lines = lines[i:i+3]  # Next 3 lines
                
                if prompt_lines and completion_lines:
                    prompt = '\n'.join(prompt_lines)
                    completion = '\n'.join(completion_lines)
                    
                    completion_examples.append({
                        'type': 'code_completion',
                        'language': language,
                        'instruction': f"Complete the next few lines of this {language} code:\n\n{prompt}",
                        'response': completion,
                        'metadata': {
                            'file_path': file_data['file_path'],
                        }
                    })
    
    return completion_examples


def generate_function_explanation_examples(analyzed_files: List[Dict]) -> List[Dict]:
    explanation_examples = []
    
    for file_data in analyzed_files:
        language = file_data['language']
        symbols = file_data.get('symbols', [])
        
        for symbol in symbols:
            if symbol['type'] in ['function', 'class'] and symbol.get('docstring'):
                # Create explanation example using docstring
                code = symbol['code']
                docstring = symbol['docstring']
                
                # Remove the docstring from the code to avoid information leakage
                code_without_docstring = re.sub(r'""".*?"""', '"""DOCSTRING_PLACEHOLDER"""', code, flags=re.DOTALL)
                
                explanation_examples.append({
                    'type': 'function_explanation',
                    'language': language,
                    'instruction': f"Explain what this {language} {symbol['type']} does:\n\n{code_without_docstring}",
                    'response': f"This {symbol['type']} `{symbol['name']}` {docstring}",
                    'metadata': {
                        'file_path': file_data['file_path'],
                        'symbol_name': symbol['name'],
                    }
                })
    
    return explanation_examples


def generate_code_commenting_examples(analyzed_files: List[Dict]) -> List[Dict]:
    commenting_examples = []
    
    for file_data in analyzed_files:
        language = file_data['language']
        symbols = file_data.get('symbols', [])
        
        for symbol in symbols:
            if symbol['type'] == 'function' and len(symbol['code'].splitlines()) >= 5:
                code = symbol['code']
                
                # Remove existing comments and docstrings
                if language == 'python':
                    # Remove docstrings
                    code_without_comments = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
                    # Remove single-line comments
                    code_without_comments = re.sub(r'#.*', '', code_without_comments)
                elif language in ['javascript', 'typescript', 'java']:
                    # Remove block comments
                    code_without_comments = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
                    # Remove single-line comments
                    code_without_comments = re.sub(r'//.*', '', code_without_comments)
                else:
                    code_without_comments = code
                
                # If there are comments or docstrings (the code changed after removing them)
                if code != code_without_comments:
                    commenting_examples.append({
                        'type': 'code_commenting',
                        'language': language,
                        'instruction': f"Add appropriate comments to this {language} code:\n\n{code_without_comments}",
                        'response': code,
                        'metadata': {
                            'file_path': file_data['file_path'],
                            'symbol_name': symbol['name'],
                        }
                    })
    
    return commenting_examples


def generate_qa_examples(analyzed_files: List[Dict], doc_contents: List[Dict]) -> List[Dict]:
    qa_examples = []
    
    # Extract repository name from the first file path
    if analyzed_files:
        repo_path = Path(analyzed_files[0]['file_path'])
        repo_parts = repo_path.parts
        repo_name = repo_parts[1] if len(repo_parts) > 1 else "the repository"
    else:
        repo_name = "the repository"
    
    # Generate Q&A from README files
    for doc in doc_contents:
        content = doc['content']
        if not content:
            continue
        
        # Simple example for the repository's purpose
        qa_examples.append({
            'type': 'qa',
            'instruction': f"What is the purpose of {repo_name}?",
            'response': f"Based on the documentation, {content[:500]}...",  # First part of the README
            'metadata': {
                'file_path': doc['file_path'],
                'doc_type': 'readme',
            }
        })
        
        # If README has sections (markdown headers)
        sections = re.findall(r'#{1,6}\s+(.+)', content)
        if sections:
            qa_examples.append({
                'type': 'qa',
                'instruction': f"What are the main sections of the {repo_name} documentation?",
                'response': f"The main sections in the documentation are: " + ", ".join(sections[:5]),
                'metadata': {
                    'file_path': doc['file_path'],
                    'doc_type': 'readme_sections',
                }
            })
    
    # Generate Q&A about code structure
    if analyzed_files:
        # Extract file types
        file_types = {}
        for file in analyzed_files:
            lang = file['language']
            if lang != 'unknown':
                file_types[lang] = file_types.get(lang, 0) + 1
        
        if file_types:
            file_types_str = ", ".join([f"{count} {lang} files" for lang, count in file_types.items()])
            qa_examples.append({
                'type': 'qa',
                'instruction': f"What programming languages are used in {repo_name}?",
                'response': f"This repository contains {file_types_str}.",
                'metadata': {
                    'doc_type': 'code_structure',
                }
            })
        
        # Extract main functions/classes
        main_symbols = []
        for file in analyzed_files:
            for symbol in file.get('symbols', []):
                main_symbols.append({
                    'name': symbol['name'],
                    'type': symbol['type'],
                    'file': os.path.basename(file['file_path'])
                })
        
        if main_symbols:
            # Limit to avoid overly long examples
            main_symbols = main_symbols[:10]
            symbols_str = ", ".join([f"{s['name']} ({s['type']} in {s['file']})" for s in main_symbols])
            qa_examples.append({
                'type': 'qa',
                'instruction': f"What are the main functions or classes in {repo_name}?",
                'response': f"Some of the main functions and classes include: {symbols_str}.",
                'metadata': {
                    'doc_type': 'code_symbols',
                }
            })
    
    return qa_examples


def generate_chat_examples(analyzed_files: List[Dict], doc_contents: List[Dict]) -> List[Dict]:
    chat_examples = []
    
    # Extract repository name
    if analyzed_files:
        repo_path = Path(analyzed_files[0]['file_path'])
        repo_parts = repo_path.parts
        repo_name = repo_parts[1] if len(repo_parts) > 1 else "the repository"
    else:
        repo_name = "the repository"
    
    # Generate chat about code usage
    for file in analyzed_files:
        for symbol in file.get('symbols', []):
            if symbol['type'] == 'function' and symbol.get('docstring'):
                chat_examples.append({
                    'type': 'chat',
                    'messages': [
                        {"role": "system", "content": f"You are an assistant that helps users understand and use code from {repo_name}."},
                        {"role": "user", "content": f"How do I use the {symbol['name']} function?"},
                        {"role": "assistant", "content": f"The `{symbol['name']}` function is defined as follows:\n\n```{file['language']}\n{symbol['code']}\n```\n\n{symbol['docstring']}\n\nTo use this function, you would call it with appropriate parameters as shown in the signature."},
                    ],
                    'metadata': {
                        'file_path': file['file_path'],
                        'symbol_name': symbol['name'],
                    }
                })
    
    # Generate chat about repository overview
    if doc_contents:
        readme_content = ""
        for doc in doc_contents:
            if "readme" in doc['file_path'].lower():
                readme_content = doc['content']
                break
        
        if readme_content:
            # Trim to a reasonable length
            if len(readme_content) > 1000:
                readme_summary = readme_content[:1000] + "..."
            else:
                readme_summary = readme_content
                
            chat_examples.append({
                'type': 'chat',
                'messages': [
                    {"role": "system", "content": f"You are an assistant that helps users understand {repo_name}."},
                    {"role": "user", "content": f"Can you give me an overview of this repository?"},
                    {"role": "assistant", "content": f"Based on the README:\n\n{readme_summary}"},
                    {"role": "user", "content": "How would I get started with this code?"},
                    {"role": "assistant", "content": "To get started with this repository, I would recommend the following steps:\n\n1. Clone the repository\n2. Review the README for installation instructions\n3. Look at the main modules and functions to understand the structure\n4. Check for example usage in documentation or tests"}
                ],
                'metadata': {
                    'file_path': 'README',
                }
            })
    
    return chat_examples


def format_training_examples(examples: List[Dict], model_format: str = "chatml") -> List[Dict]:
    formatted_examples = []
    
    for example in examples:
        if model_format == "chatml":
            # Format for chat models (ChatML format)
            if example['type'] == 'chat':
                # Chat examples are already in the right format
                formatted_examples.append(example['messages'])
            else:
                # Convert instruction-response to chat format
                formatted_examples.append([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['response']}
                ])
        
        elif model_format == "instruction":
            # Format for instruction models
            if example['type'] == 'chat':
                # Convert chat to instruction format
                messages = example['messages']
                instruction = ""
                for i, msg in enumerate(messages):
                    if msg['role'] == 'user':
                        instruction += f"{msg['role']}: {msg['content']}\n\n"
                        if i + 1 < len(messages) and messages[i+1]['role'] == 'assistant':
                            formatted_examples.append({
                                "instruction": instruction.strip(),
                                "response": messages[i+1]['content']
                            })
                            instruction = ""
            else:
                # Instruction-response already in right format
                formatted_examples.append({
                    "instruction": example['instruction'],
                    "response": example['response']
                })
        
        else:
            # Default format (preserve original)
            formatted_examples.append(example)
    
    return formatted_examples


def filter_examples(examples: List[Dict]) -> List[Dict]:
    # Filter out examples with very short responses
    filtered = [ex for ex in examples if len(ex.get('response', '')) >= 10]
    
    # Simple deduplication based on instruction text
    seen_instructions = set()
    deduplicated = []
    
    for ex in filtered:
        instruction = ex.get('instruction', '')
        if instruction and instruction not in seen_instructions:
            seen_instructions.add(instruction)
            deduplicated.append(ex)
    
    return deduplicated


def create_repository_dataset(repo_url: str, output_dir: str, model_format: str = "chatml"):
    logger.info(f"Creating dataset from repository: {repo_url}")
    
    # Step 1: Clone the repository
    repo_path = clone_repository(repo_url)
    
    # Step 2: Extract repository metadata
    repo_metadata = extract_metadata(repo_path)
    logger.info(f"Repository metadata: {repo_metadata}")
    
    # Step 3: Parse directory structure
    dir_structure = parse_directory_structure(repo_path)
    
    # Step 4: Filter relevant files
    code_files = filter_code_files(dir_structure, repo_path)
    doc_files = filter_documentation_files(dir_structure, repo_path)
    
    logger.info(f"Found {len(code_files)} code files and {len(doc_files)} documentation files")
    
    # Step 5: Process files
    processed_code_files = []
    for file_path in tqdm(code_files, desc="Processing code files"):
        file_data = process_file(file_path)
        if file_data['content']:
            processed_code_files.append(file_data)
    
    processed_doc_files = []
    for file_path in tqdm(doc_files, desc="Processing documentation files"):
        file_data = process_file(file_path)
        if file_data['content']:
            processed_doc_files.append(file_data)
    
    logger.info(f"Successfully processed {len(processed_code_files)} code files and {len(processed_doc_files)} documentation files")
    
    # Step 6: Analyze code files
    analyzed_code_files = []
    for file_data in tqdm(processed_code_files, desc="Analyzing code files"):
        analyzed_file = analyze_code_file(file_data)
        analyzed_code_files.append(analyzed_file)
    
    # Step 7: Generate training examples
    logger.info("Generating training examples...")
    
    # Code completion examples
    completion_examples = generate_code_completion_examples(analyzed_code_files)
    logger.info(f"Generated {len(completion_examples)} code completion examples")
    
    # Function explanation examples
    explanation_examples = generate_function_explanation_examples(analyzed_code_files)
    logger.info(f"Generated {len(explanation_examples)} function explanation examples")
    
    # Code commenting examples
    commenting_examples = generate_code_commenting_examples(analyzed_code_files)
    logger.info(f"Generated {len(commenting_examples)} code commenting examples")
    
    # Q&A examples
    qa_examples = generate_qa_examples(analyzed_code_files, processed_doc_files)
    logger.info(f"Generated {len(qa_examples)} Q&A examples")
    
    # Chat examples
    chat_examples = generate_chat_examples(analyzed_code_files, processed_doc_files)
    logger.info(f"Generated {len(chat_examples)} chat examples")
    
    # Combine all examples
    all_examples = (
        completion_examples +
        explanation_examples +
        commenting_examples +
        qa_examples +
        chat_examples
    )
    logger.info(f"Total number of examples before filtering: {len(all_examples)}")
    
    # Filter and format examples
    filtered_examples = filter_examples(all_examples)
    logger.info(f"Number of examples after filtering: {len(filtered_examples)}")
    
    formatted_examples = format_training_examples(filtered_examples, model_format)
    logger.info(f"Formatted {len(formatted_examples)} examples for {model_format}")
    
    # Step 8: Save dataset
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each example type separately
    dataset_files = {}
    
    for example_type, examples in {
        'code_completion': completion_examples,
        'function_explanation': explanation_examples,
        'code_commenting': commenting_examples,
        'qa': qa_examples,
        'chat': chat_examples
    }.items():
        if examples:
            output_file = os.path.join(output_dir, f"{example_type}_examples.jsonl")
            with open(output_file, 'w') as f:
                for example in examples:
                    f.write(json.dumps(example) + '\n')
            dataset_files[example_type] = output_file
    
    # Save combined dataset
    combined_output = os.path.join(output_dir, "combined_dataset.jsonl")
    with open(combined_output, 'w') as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save metadata
    metadata = {
        'repository': repo_metadata,
        'generation_date': pd.Timestamp.now().isoformat(),
        'statistics': {
            'code_files': len(processed_code_files),
            'doc_files': len(processed_doc_files),
            'total_examples': len(formatted_examples),
            'example_types': {
                'code_completion': len(completion_examples),
                'function_explanation': len(explanation_examples),
                'code_commenting': len(commenting_examples),
                'qa': len(qa_examples),
                'chat': len(chat_examples)
            }
        },
        'model_format': model_format,
        'dataset_files': dataset_files
    }
    
    metadata_file = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset created successfully at {output_dir}")
    return combined_output