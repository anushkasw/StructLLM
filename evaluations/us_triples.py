import argparse
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from rel_verbaliser import get_rel2prompt
from data_loader import get_RC_data, get_JRE_data
from embeddings import EMBD_processor


class US_calculator:
    def __init__(self, phi, embedder):
        self.phi = phi
        self.embedder = embedder

    def calculate_uniqueness(self, vectors):
        """Calculate the Uniqueness Score using cosine similarity and a threshold."""
        similarity_matrix = cosine_similarity(vectors)
        np.fill_diagonal(similarity_matrix, 1)  # Ignore self-similarity

        # Count pairs with cosine similarity smaller than the threshold
        count_smaller_than_phi = np.sum(similarity_matrix < self.phi)

        total_pairs = len(vectors) * (len(vectors) - 1)
        return count_smaller_than_phi / total_pairs if total_pairs > 0 else 1

    def calculate_uniqueness_for_text(self, triples):
        """Calculate the uniqueness score for a batch of texts."""
        vectors = []
        for triple in triples:
            try:
                vectors.append(self.embedder.get_per_triple_embds(triple))
            except:
                continue

        try:
            return self.calculate_uniqueness(np.array(vectors))
        except:
            return 1

    def process_triples(self, triples):
        # if no triples, return 0
        if not triples or len(triples) == 0:
            return 1
        return self.calculate_uniqueness_for_text(triples)


def calculate_uniqueness_score(tmp_dict, embedder, output_all_scores=False):
    """Calculate the Uniqueness Score for a dataset using multi-threading."""
    scores = []
    ids = []
    for dict_ in tmp_dict:
        ids.append(dict_['id'])

    us_dict = {key: {} for key in ids}
    for phi in [0.85, 0.9, 0.95, 0.96, 0.97, 0.98]:
        key = f'us_{phi}'
        US = US_calculator(phi, embedder)
        for dict_ in tmp_dict:
            triples = dict_['pred_label']
            us = US.process_triples(triples)
            scores.append(us)
            us_dict[dict_['id']][key] = us
    return us_dict


def main(args):
    embdder = EMBD_processor(args)

    data = args.dataset

    # if args.exp=='plm':
    #     models = ["PLM/UniRel", "PLM/RIFRE", "PLM/SPN4RE", "PLM/TDEER"]
    # else:
    #     models = ["OpenAI/gpt-4o", "openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #               "mistralai/Mistral-Nemo-Instruct-2407", "google/gemma-2-9b-it"]

    model = args.model
    files = list(
        Path(f'{args.res_path}/JRE/{args.exp}/{data}/{model}'
                ).rglob('*.jsonl'))

    for file in tqdm(files):
        if not file.name.startswith(('cs', 'us', 'ts', 'prf', 'fs')):
            outfile = file.with_name(f'us_{file.name}')
            if not os.path.exists(outfile) or args.overwrite:
                data_dict = []
                with open(file, "r") as f:
                    for line in f.read().splitlines():
                        sample = json.loads(line)
                        data_dict.append(sample)
                        # data_dict[sample['id']] = sample
                us_dict = calculate_uniqueness_score(data_dict, embdder)

                with open(outfile, "w") as f:
                    json.dump(us_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="RC")
    parser.add_argument('--dataset', '-d', type=str, required=False, help="Dataset Type", default="NYT10")
    parser.add_argument('--model', '-m', type=str, required=True, help="model")
    parser.add_argument("--overwrite", action='store_true', help="if use huggingface pipeline")

    parser.add_argument('--base_path', type=str, required=True)
    parser.add_argument('--res_path', '-dir', type=str, required=True)
    parser.add_argument("--embd_path", type=str,
                        required=True,
                        help="raw data dir")
    parser.add_argument("--out_path", type=str,
                        required=True,
                        help="raw data dir")

    args = parser.parse_args()
    main(args)
