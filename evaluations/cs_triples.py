import argparse
import os
import json

import numpy as np
from tqdm import tqdm

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from embeddings import EMBD_processor


def calculate_completeness_score(tmp_dict, embedder, model_name=None, threshold=None,
                                 output_all_scores=False):
    gt_triples = []
    pred_triples = []
    ids = []
    for dict_ in tmp_dict:
        ids.append(dict_['id'])
        gt_triples.extend(dict_['true_label'])
        pred_triples.extend(dict_['pred_label'])

    gt_triple_emb_store, gt_relation_emb_store = embedder.get_triple_embds(gt_triples)
    pred_triple_emb_store, pred_relation_emb_store = embedder.get_triple_embds(pred_triples)

    cs_dict = {key: {} for key in ids}
    for threshold in [0.85, 0.9, 0.95, 0.96, 0.97, 0.98]:
        # completeness_scores = []
        key = f'cs_{threshold}'
        for dict_ in tmp_dict:
            gt_triples = dict_['true_label']
            pred_triples = dict_['pred_label']

            if not pred_triples or len(pred_triples) == 0:
                cs_dict[dict_['id']][key] = 0
                # completeness_scores.append(0)
                continue

            if len(gt_triples) == 0:
                cs_dict[dict_['id']][key] = 1
                # completeness_scores.append(1)
                continue

            gt_embeddings = {str(triple): gt_triple_emb_store[str(triple)] for triple in gt_triples}
            gt_recalls = {gt_triple: 0 for gt_triple in gt_embeddings.keys()}

            extracted_triple_embeddings = []
            for triple in pred_triples:
                try:
                    extracted_triple_embeddings.append(pred_triple_emb_store[str(triple)])
                except:
                    continue
            if len(extracted_triple_embeddings) == 0:
                cs_dict[dict_['id']][key] = -1
                continue
            for gt_triple, gt_embedding in gt_embeddings.items():
                similarity_scores = cosine_similarity([gt_embedding], extracted_triple_embeddings)
                best_match_score = np.max(similarity_scores)
                # best_match_index = np.argmax(similarity_scores)
                if best_match_score >= threshold:
                        gt_recalls[gt_triple] = 1

            # Compute completeness score for this text
            score = sum(gt_recalls.values()) / len(gt_recalls) if len(gt_recalls) > 0 else 0
            # completeness_scores.append(score)
            cs_dict[dict_['id']][key] = score

        # avg_completeness_score = np.mean(completeness_scores) if completeness_scores else 0

    return cs_dict


def main(args):
    embdder = EMBD_processor(args)
    data = args.dataset
    model = args.model
    # if args.exp=='plm':
    #     models = ["PLM/UniRel", "PLM/RIFRE", "PLM/SPN4RE", "PLM/TDEER"]
    # else:
    #     models = ["openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #               "mistralai/Mistral-Nemo-Instruct-2407", "google/gemma-2-9b-it",
    #               'OpenAI/gpt-4o']

    # for model in models:
    files = list(
        Path(f'{args.res_path}/JRE/{args.exp}/{data}/{model}'
                ).rglob('*.jsonl'))

    for file in tqdm(files):
        if not file.name.startswith(('cs', 'us', 'ts', 'prf', 'fs')):
            outfile = file.with_name(f'cs_{file.name}')
            if not os.path.exists(outfile) or args.overwrite:
                data_dict = []
                with open(file, "r") as f:
                    for line in f.read().splitlines():
                        sample = json.loads(line)
                        data_dict.append(sample)
                        # data_dict[sample['id']] = sample
                cs_dict = calculate_completeness_score(data_dict, embdder)

                with open(outfile, "w") as f:
                    json.dump(cs_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="results_2stage")
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
