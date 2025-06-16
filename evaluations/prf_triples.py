import argparse
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from rel_verbaliser import get_rel2prompt
from data_loader import get_RC_data, get_JRE_data


def calculate_jre_metrics(predicted_relations, ground_truth_relations):
        prec, rec = 0, 0

        # Count correct predictions
        for pred in predicted_relations:
            if pred in ground_truth_relations:
                prec += 1

        for gt in ground_truth_relations:
            if gt in predicted_relations:
                rec += 1

        precision = prec / len(predicted_relations)
        recall = rec / len(ground_truth_relations)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return precision, recall, f1


def f1_score(true, pred_result):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            if golden not in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable', 'NONE']:
                correct_positive += 1
        if golden not in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable', 'NONE']:
            gold_positive +=1
        if pred_result[i] not in ['NA', 'na', 'no_relation', 'Other', 'Others', 'false', 'unanswerable', 'NONE']:
            pred_positive += 1
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    return micro_p, micro_r, micro_f1


def f1_score_na(true, pred_result):
    correct = 0
    total = len(true)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    for i in range(total):
        golden = true[i]
        if golden == pred_result[i]:
            correct += 1
            correct_positive += 1
        gold_positive +=1
        pred_positive += 1
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0
    return micro_p, micro_r, micro_f1


def get_traditional_scores(exp, tmp_dict, prompt2rel):
    if exp != 'RC':
        ids = []
        for dict_ in tmp_dict:
            ids.append(dict_['id'])
            triples = dict_['true_label']
            for trip in triples:
                if len(trip) > 1:
                    if trip[1] in prompt2rel:
                        trip[1] = prompt2rel[trip[1]]

            triples = dict_['pred_label']
            for trip in triples:
                if len(trip) > 1:
                    if trip[1] in prompt2rel:
                        trip[1] = prompt2rel[trip[1]]

        prf_dict = {key: {} for key in ids}
        for dict_ in tmp_dict:
            try:
                pred_triple_str = [" ".join(triple) for triple in dict_['pred_label']]
                gt_triple_str = [" ".join(triple) for triple in dict_['true_label']]
                p, r, f = calculate_jre_metrics(pred_triple_str, gt_triple_str)
                prf_dict[dict_['id']]['p'] = p
                prf_dict[dict_['id']]['r'] = r
                prf_dict[dict_['id']]['f'] = f
            except:
                continue
        return prf_dict
    else:
        true_label = []
        pred_label = []
        for idx, dict_ in tmp_dict.items():
            relation = dict_['pred_label']
            if relation in prompt2rel:
                pred_label.append(prompt2rel[relation])
            else:
                pred_label.append(relation)
            true_label.append(dict_['true_label'])
        p, r, f = f1_score(pred_label, true_label)
        return p, r, f


def main(args):
    data = args.dataset
    data_dict, rel2id = get_JRE_data(data, args.base_path)

    rel2prompt = get_rel2prompt(data, rel2id)
    prompt2rel = {val: key for key, val in rel2prompt.items()}

    if args.exp=='plm':
        models = ["PLM/UniRel", "PLM/RIFRE", "PLM/SPN4RE", "PLM/TDEER"]
    else:
        models = ["OpenAI/gpt-4o", "openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                  "mistralai/Mistral-Nemo-Instruct-2407", "google/gemma-2-9b-it"]

    for model in models:
        files = list(
            Path(f'{args.res_path}/JRE/{args.exp}/{data}/{model}'
                 ).rglob('*.jsonl'))

        for file in tqdm(files):
            if not file.name.startswith(('cs', 'us', 'ts', 'prf', 'fs')):
                outfile = file.with_name(f'prf_{file.name}')
                if not os.path.exists(outfile) or args.overwrite:
                    data_dict = []
                    with open(file, "r") as f:
                        for line in f.read().splitlines():
                            sample = json.loads(line)
                            data_dict.append(sample)

                    prf_dict = get_traditional_scores(args.exp, data_dict, prompt2rel)
                    with open(outfile, "w") as f:
                        json.dump(prf_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="JRE")
    parser.add_argument('--dataset', '-d', type=str, required=False, help="Dataset Type", default="NYT10")

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
