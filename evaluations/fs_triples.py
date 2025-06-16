import argparse
import os
import json

import pandas as pd
from tqdm import tqdm

from pathlib import Path

def calculate_factualness_score(tmp_dict, fs_path):
    fact_df = pd.read_json(fs_path, lines=True)
    fact_df['sent_id'] = fact_df.apply(lambda x: x['custom_id'].split('_')[0], axis=1)
    fact_df['triple_id'] = fact_df.apply(lambda x: x['custom_id'].split('_')[1], axis=1)
    fact_df['decision'] = fact_df.apply(lambda x: x['response']['body']['choices'][0]['message']['content'], axis=1)

    grouped = fact_df.groupby('sent_id').agg(
        true_count=('decision', lambda x: (x == 'true').sum()),
        total_count=('decision', 'count')
    )

    # Calculate the proportion of true values
    grouped['true_ratio'] = grouped['true_count'] / grouped['total_count']
    result_dict = grouped.reset_index().set_index('sent_id').to_dict('index')

    fs_dict = {}
    for key in tmp_dict:
        if key in result_dict:
            fs_dict[key] = result_dict[key]['true_ratio']
        else:
            fs_dict[key] = 0

    return fs_dict


def get_info(exp, file):
    parts = file.parts
    struct, prompt, demo, seed, k = None, None, None, None, None
    if exp=='plm':
        seed = file.parts[-1].split('.')[0][-1]
    elif exp=='1stage':
        prompt = parts[-1].split('_')[-1].split('-')[0]
        k = int(parts[-1].split('-')[-1].split('.')[0])
        seed = parts[-2].split('-')[-1]
        demo = parts[-3]
    elif exp=='structure_extract':
        prompt = parts[-1].split('_')[-1].split('-')[1]
        struct = parts[-1].split('-')[0]
        k = int(parts[-1].split('_')[-1].split('-')[-1].split('.')[0])
        seed = parts[-2].split('-')[-1]
        demo = parts[-3]
    else:
        prompt = parts[-1].split('-')[0]
        k = int(parts[-1].split('_')[-1].split('-')[-1].split('.')[0])
        seed = parts[-2].split('-')[-1]
        demo = parts[-3]
    return struct, prompt, seed, demo, k


def main(args):
    data = args.dataset
    if args.exp=='plm':
        models = ["PLM/UniRel", "PLM/RIFRE", "PLM/SPN4RE", "PLM/TDEER"]
    else:
        models = ["OpenAI/gpt-4o", "openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                  "mistralai/Mistral-Nemo-Instruct-2407", "google/gemma-2-9b-it"]

    missing_model_dict = {key:0 for key in models}

    for model in models:
        files = list(
            Path(f'{args.res_path}/JRE/{args.exp}/{data}/{model}'
                 ).rglob('*.jsonl'))

        for file in tqdm(files):
            if not file.name.startswith(('cs', 'us', 'ts', 'prf', 'fs')):
                outfile = file.with_name(f'fs_{file.name}')
                if not os.path.exists(outfile) or args.overwrite:
                    struct, prompt, seed, demo, k = get_info(args.exp, file)

                    if demo in ['fast_votek', 'random', 'knn'] or k in [10, 20] or seed in ['100', '3', '4', '5'] or parser=='stanza':
                        continue

                    if args.exp=='plm':
                        fs_path = f'{args.res_path}/factual_batches/JRE/{args.exp}/{data}/{model}/None/seed-{seed}/output/test_results_{seed}.jsonl'
                    else:
                        if struct:
                            fs_path = f'{args.res_path}/factual_batches/JRE/{args.exp}/{data}/{model}/{demo}/seed-{seed}/output/{struct}-{prompt}-{k}.jsonl'
                        else:
                            fs_path = f'{args.res_path}/factual_batches/JRE/{args.exp}/{data}/{model}/{demo}/seed-{seed}/output/{prompt}-{k}.jsonl'

                    if os.path.exists(fs_path):
                        data_dict = {}
                        with open(file, "r") as f:
                            for line in f.read().splitlines():
                                sample = json.loads(line)
                                data_dict[sample['id']] = sample
                        fs_dict = calculate_factualness_score(data_dict, fs_path)

                        with open(outfile, "w") as f:
                            json.dump(fs_dict, f)
                    else:
                        print(f'Factual response absent at {fs_path}!')
                        missing_model_dict[model]+=1
                        continue
    print(missing_model_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=True, help="Hugging Face API access token")
    parser.add_argument('--exp', '-e', type=str, required=False, help="Experiment Type", default="results_2stage")
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
