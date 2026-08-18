from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

print("Downloading Gujarati metadata/file...")

file_path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/gujtrain.parquet",
    repo_type="dataset",
)

print("File downloaded to:")
print(file_path)

table = pq.ParquetFile(file_path)

print("\nNumber of row groups:", table.num_row_groups)
print("Schema:")
print(table.schema)

print("\nFirst rows:")

df = table.read_row_group(0).to_pandas()

print(df.head(3).to_string())