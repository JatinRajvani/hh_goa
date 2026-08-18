from datasets import load_dataset

print("Loading MSMARCO-XI...")

dataset = load_dataset(
    "ai4bharat/MSMARCO-XI",
    split="train",
    streaming=True,
)

print("Dataset stream created.")

for i, example in enumerate(dataset):
    print(f"\n========== Example {i + 1} ==========")

    print("Keys:")
    print(example.keys())

    print("\nExample:")
    print(example)

    if i == 2:
        break