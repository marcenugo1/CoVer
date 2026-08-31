import argparse
from huggingface_hub import hf_hub_download
import shutil


parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=["CodeContests_train","LiveBench","LiveCodeBench","CodeContests","CodeForces","MBPP-ReasonFlux"],
    required=True,
    help="Which dataset to download"
)
args = parser.parse_args()
dataset = args.dataset


if dataset == "CodeContests_train":
    split = "train"
else:
    split = "test"


REMOTE_FILENAME = {"MBPP-ReasonFlux": "MBPP"}
remote_stem = REMOTE_FILENAME.get(dataset, dataset)


cached_path = hf_hub_download(
    repo_id=f"Gen-Verse/{dataset}",
    repo_type="dataset",
    filename=f"{split}/{remote_stem}.json"
)
shutil.copy(cached_path, f"./{dataset}.json")
