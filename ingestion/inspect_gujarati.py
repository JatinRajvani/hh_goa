from datasets import load_dataset

print("Loading Gujarati MSMARCO-XI shard...")

dataset = load_dataset(
    "parquet",
    data_files="hf://datasets/ai4bharat/MSMARCO-XI/train/gujtrain.parquet",
    split="train",
    streaming=True,
)

print("Stream created!")

for i, example in enumerate(dataset):
    print(f"\n========== Example {i + 1} ==========")

    print("\nTop-level fields:")
    print(example.keys())

    print("\nFull example:")
    print(example)

    if i == 0:
        break