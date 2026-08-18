from huggingface_hub import HfApi

api = HfApi()

repo = api.dataset_info("ai4bharat/MSMARCO-XI")

for file in repo.siblings:
    if file.rfilename in [
        "train/gujtrain.parquet",
        "train/hintrain.parquet",
        "validation/gujval.parquet",
        "validation/hinval.parquet",
    ]:
        print(file.rfilename)
        print("Size:", file.size)
        print()