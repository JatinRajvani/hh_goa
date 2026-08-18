import os
import json
import hashlib
import re
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

# ==========================================
# Configuration
# ==========================================
REPO_ID = "ai4bharat/MSMARCO-XI"
FILENAME = "train/gujtrain.parquet"

# Extraction Limits for development subset
MAX_RECORDS = 5000
MAX_PASSAGES = 50000

# Output paths (must match layout and phase specifications)
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "english_passages.jsonl")
EVAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_queries.jsonl")

# Processing batch size for pyarrow streaming
BATCH_SIZE = 500


def clean_text(text):
    """Normalize whitespace and clean text without rewriting factual content."""
    if not text:
        return ""
    # Decode bytes if needed
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    # Collapse multiple whitespaces and strip
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_data():
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading/loading file '{FILENAME}' from repository '{REPO_ID}'...")
    try:
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
        )
        print(f"File located at: {file_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

    print("Opening Parquet file...")
    parquet_file = pq.ParquetFile(file_path)
    
    print(f"Total row groups in file: {parquet_file.num_row_groups}")
    
    # Initialize statistics
    stats = {
        "records_processed": 0,
        "passages_encountered": 0,
        "passages_written": 0,
        "empty_skipped": 0,
        "duplicates_detected": 0,
        "selected_passages": 0,
        "non_selected_passages": 0,
        "unique_queries": 0,
    }
    
    # Set to keep track of passage text hashes to detect exact text duplicates
    seen_passage_hashes = set()
    
    # Open output files
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out, \
         open(EVAL_OUTPUT_FILE, "w", encoding="utf-8") as f_eval:
        
        # Stream batches of records
        batch_iter = parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["query_id", "Eng_Query", "Eng_Answer", "query_type", "passages", "source_lang", "target_lang"]
        )
        
        limit_reached = False
        
        for batch_idx, batch in enumerate(batch_iter):
            if limit_reached:
                break
                
            records = batch.to_pylist()
            
            for record in records:
                if stats["records_processed"] >= MAX_RECORDS:
                    limit_reached = True
                    break
                if stats["passages_written"] >= MAX_PASSAGES:
                    limit_reached = True
                    break
                
                query_id = record.get("query_id")
                eng_query = clean_text(record.get("Eng_Query"))
                query_type = record.get("query_type")
                source_lang = record.get("source_lang")
                target_lang = record.get("target_lang")
                passages_struct = record.get("passages")
                
                if not query_id or not eng_query:
                    continue
                
                stats["records_processed"] += 1
                
                relevant_passage_ids = []
                
                if passages_struct:
                    eng_passages = passages_struct.get("English_passages", [])
                    is_selected_list = passages_struct.get("is_selected", [])
                    
                    stats["passages_encountered"] += len(eng_passages)
                    
                    for idx, passage_text in enumerate(eng_passages):
                        if stats["passages_written"] >= MAX_PASSAGES:
                            limit_reached = True
                            break
                            
                        cleaned_passage = clean_text(passage_text)
                        
                        # Skip empty or trivial passages
                        if not cleaned_passage or len(cleaned_passage) < 3:
                            stats["empty_skipped"] += 1
                            continue
                            
                        # Duplicate check (exact text duplicate using MD5 hash)
                        text_hash = hashlib.md5(cleaned_passage.encode("utf-8")).hexdigest()
                        if text_hash in seen_passage_hashes:
                            stats["duplicates_detected"] += 1
                            # Continue to index/write it so we preserve query-passage relations,
                            # but we count it in stats. Alternatively, skip. Let's write it to keep complete indexing,
                            # or follow typical IR corpus deduplication. In IR evaluation, keeping it is standard unless
                            # globally deduplicating. The prompt says "report duplicate count before deciding how aggressively to deduplicate".
                            # Thus, we preserve and write them, but track the count.
                        else:
                            seen_passage_hashes.add(text_hash)
                            
                        is_selected = 0
                        if idx < len(is_selected_list):
                            is_selected = int(is_selected_list[idx])
                            
                        doc_id = f"{query_id}_{idx}"
                        
                        if is_selected == 1:
                            stats["selected_passages"] += 1
                            relevant_passage_ids.append(doc_id)
                        else:
                            stats["non_selected_passages"] += 1
                            
                        doc_record = {
                            "document_id": doc_id,
                            "text": cleaned_passage,
                            "metadata": {
                                "query_id": query_id,
                                "passage_index": idx,
                                "is_selected": is_selected,
                                "query_type": query_type,
                                "source_lang": source_lang,
                                "target_lang": target_lang
                            }
                        }
                        
                        f_out.write(json.dumps(doc_record, ensure_ascii=False) + "\n")
                        stats["passages_written"] += 1
                
                # Write evaluation query (even if no passages were found/selected, though standard MSMARCO queries have selections)
                eval_record = {
                    "query_id": query_id,
                    "query": eng_query,
                    "relevant_passage_ids": relevant_passage_ids
                }
                f_eval.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                stats["unique_queries"] += 1
                
            print(f"Processed batch {batch_idx + 1}... Records: {stats['records_processed']}, Passages written: {stats['passages_written']}")
            
    print("\n========== Extraction Complete ==========")
    for key, value in stats.items():
        # Print stats nicely formatted
        display_name = key.replace("_", " ").capitalize()
        print(f"{display_name:<25}: {value:,}")
        
    print(f"\nOutputs written to:\n  - Corpus: {OUTPUT_FILE}\n  - Evaluation queries: {EVAL_OUTPUT_FILE}\n")
    return True


def validate_outputs():
    print("========== Starting Validation ==========")
    
    if not os.path.exists(OUTPUT_FILE) or not os.path.exists(EVAL_OUTPUT_FILE):
        print("Error: Output files do not exist.")
        return False
        
    # 1. Validate English Passages Corpus
    doc_ids = set()
    doc_count = 0
    empty_text_count = 0
    missing_fields = 0
    
    print(f"Validating corpus file: {OUTPUT_FILE}...")
    sample_docs = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc_count += 1
            try:
                doc = json.loads(line)
                doc_id = doc.get("document_id")
                text = doc.get("text")
                metadata = doc.get("metadata", {})
                
                # Check required fields
                if not doc_id or not text or "query_id" not in metadata or "passage_index" not in metadata or "is_selected" not in metadata:
                    missing_fields += 1
                
                # Check for uniqueness
                if doc_id:
                    if doc_id in doc_ids:
                        print(f"Warning: Duplicate document_id found: {doc_id}")
                    doc_ids.add(doc_id)
                    
                # Check for empty text
                if not text or len(text.strip()) == 0:
                    empty_text_count += 1
                    
                # Keep 3 samples
                if len(sample_docs) < 3:
                    sample_docs.append(doc)
            except Exception as e:
                print(f"Error parsing line {i+1}: {e}")
                return False
                
    print(f"  - Total documents read: {doc_count:,}")
    print(f"  - Unique document IDs: {len(doc_ids):,}")
    print(f"  - Missing fields count: {missing_fields}")
    print(f"  - Empty text count: {empty_text_count}")
    
    # 2. Validate Evaluation Queries
    eval_count = 0
    missing_eval_fields = 0
    dangling_references = 0
    
    print(f"\nValidating evaluation queries file: {EVAL_OUTPUT_FILE}...")
    sample_evals = []
    with open(EVAL_OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            eval_count += 1
            try:
                eq = json.loads(line)
                q_id = eq.get("query_id")
                query = eq.get("query")
                relevant_ids = eq.get("relevant_passage_ids", [])
                
                if not q_id or not query:
                    missing_eval_fields += 1
                    
                # Verify that the references exist in our corpus
                for rid in relevant_ids:
                    if rid not in doc_ids:
                        dangling_references += 1
                        
                if len(sample_evals) < 3:
                    sample_evals.append(eq)
            except Exception as e:
                print(f"Error parsing line {i+1}: {e}")
                return False
                
    print(f"  - Total evaluation queries read: {eval_count:,}")
    print(f"  - Missing fields count: {missing_eval_fields}")
    print(f"  - Dangling references count (references to non-existent passages): {dangling_references}")
    
    # 3. Print Samples
    print("\n========== Sample Corpus Records (First 3) ==========")
    for i, doc in enumerate(sample_docs):
        print(f"\nSample {i+1}:")
        print(json.dumps(doc, indent=2))
        
    print("\n========== Sample Evaluation Queries (First 3) ==========")
    for i, eq in enumerate(sample_evals):
        print(f"\nSample {i+1}:")
        print(json.dumps(eq, indent=2))
        
    # Check criteria success
    success = (
        doc_count > 0 
        and eval_count > 0 
        and len(doc_ids) == doc_count 
        and empty_text_count == 0 
        and missing_fields == 0
        and dangling_references == 0
    )
    
    if success:
        print("\nValidation PASSED successfully! Output files conform to specification.")
    else:
        print("\nValidation FAILED. Please review errors and warnings printed above.")
    return success


if __name__ == "__main__":
    if extract_data():
        validate_outputs()
