"""
Downloading and tokenising a hf dataset on the Stability AI cluster

Example Usage:
    python prep_hf_data.py suolyer/pile_pubmed-abstracts validation EleutherAI/pythia-1b 2048
"""
import argparse
import json
import os

import datasets
from transformers import AutoTokenizer

from chemnlp.data.utils import tokenise

STRING_KEY = "text"
OUT_DIR = "/fsx/proj-chemnlp/data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Which HF dataset to collect.")
    parser.add_argument(
        "dataset_split", help="train, validation or test split to collect."
    )
    parser.add_argument("model_name", help="Which HF tokeniser model class to use.")
    parser.add_argument(
        "max_length", help="Maximum context length of the model.", type=int
    )
    args = parser.parse_args()

    # load tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        model_max_length=args.max_length,
    )
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

    raw_data = datasets.load_dataset(args.dataset_name, args.dataset_split)
    raw_data = raw_data["train"]
    words_per_sample = [len(x[STRING_KEY].split(" ")) for x in raw_data]

    # process data
    tokenised_data = raw_data.map(
        lambda batch: tokenise(batch, tokenizer, args.max_length, STRING_KEY),
        remove_columns=raw_data.column_names,
        batched=True,
        batch_size=1,
        num_proc=os.cpu_count(),
        load_from_cache_file=False,
    )

    summary_stats = {
        "total_raw_samples": raw_data.num_rows,
        "average_words_per_sample": round(sum(words_per_sample) / raw_data.num_rows, 0),
        "max_words_per_sample": max(words_per_sample),
        "min_words_per_sample": min(words_per_sample),
        "total_tokenised_samples": tokenised_data.num_rows,
        "max_context_length": args.max_length,
        "total_tokens_in_billions": round(
            args.max_length * tokenised_data.num_rows / 1e9, 4
        ),
    }
    print(summary_stats)

    # save to disk
    save_path = f"{OUT_DIR}/{args.model_name}/{args.dataset_name.replace('/', '_')}"
    tokenised_data.save_to_disk(save_path)
    with open(f"{save_path}/summary_statistics.json", "w") as f:
        f.write(json.dumps(summary_stats))
