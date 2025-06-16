from tqdm import tqdm
import argparse
import math
import random
import numpy as np
import json
import sys
import os
import traceback

import torch
import gc

gc.collect()
torch.cuda.empty_cache()

from data_loader import DataProcessor
from prompt import JRE_Prompt_Processor
from hugging_api import model_init, model_inference
from utils import check_output_validity
from gpt_api import Demo

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    k_list = [0]
    args.data_dir = f'{args.data_dir}/Data_{args.re_type}'
    for k in k_list:
        print(f'\tEvaluating Shot - {k}')
        seed_list = [13, 42, 100]

        for data_seed in seed_list:
            print(f'\tEvaluating Seed - {data_seed}')
            if not args.model.lower().startswith('gpt'):
                outpath = f'{args.out_path}/{args.re_type}/{args.exp}/{args.model}/{args.dataset}/{args.demo}/seed-{data_seed}'
            else:
                outpath = f'{args.out_path}/{args.re_type}/{args.exp}/OpenAI/{args.model}/{args.dataset}/{args.demo}/seed-{data_seed}'
            os.makedirs(outpath, exist_ok=True)

            if args.exp == 'structure_extract':
                filename = f'{outpath}/{args.parser}_{args.structure}-{args.prompt}-{k}.jsonl'
            else:
                filename = f'{outpath}/{args.prompt}-{k}.jsonl'

            print(f'Saving output to {filename}')

            data_processor = DataProcessor(args, data_seed)
            print(f'\tLoading training data')
            train_dict = data_processor.get_train_examples()  # train data
            print(f'\tLoading test data')
            test_examples = data_processor.get_test_examples()

            prompt_processor = None
            prompt_processor = JRE_Prompt_Processor(args, data_processor)

            incomplete_flag, test_completed = check_output_validity(filename, test_examples)
            if test_completed:
                if len(test_completed) == len(test_examples):
                    print(f'\tResults already processed. Terminating')
                    continue

            tokenizer, model = None, None
            if not args.debug:
                if not args.model.lower().startswith('gpt'):
                    tokenizer, model = model_init(args.model, args.cache_dir)
                else:
                    demo = Demo(
                        api_key=args.api_key,
                        engine=args.model,
                        temperature=0,
                        max_tokens=300,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        logprobs=True,
                    )

            print(f'\tNumber of GPUs available: {torch.cuda.device_count()}')

            for idx, input in tqdm(test_examples.items()):
                if incomplete_flag:
                    if input.id in test_completed:
                        continue

                prompt = prompt_processor.create_prompt(input, demo_list)

                if not args.model.lower().startswith(('OpenAI', 'gpt')):
                    try:
                        if args.model.lower().startswith(('google')):
                            result = model_inference(tokenizer, model, [prompt[-1]], max_new_tokens=300, device='cuda')
                        else:
                            result = model_inference(tokenizer, model, prompt, max_new_tokens=300, device='cuda')
                    except Exception as e:
                        print(f'\n[Error] {e}')
                        continue
                else:
                    try:
                        result = demo.process_sample(prompt)

                    except Exception as e:
                        print(f'\n[Error] {e}')
                        continue

                test_res = {
                    "id": input.id,
                    "label_pred": result,
                }

                with open(filename, 'a') as f:
                    if f.tell() > 0:  # Check if file is not empty
                        f.write('\n')
                    json.dump(test_res, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', '-key', type=str, required=False, help="Hugging Face API access token")
    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--dataset', type=str, required=True, help="Dataset Name.")
    parser.add_argument('--re_type', type=str, required=True, help="RC or JRE")
    parser.add_argument('--exp', '-e', type=str, required=True, help="Experiment Type",
                        choices=['structure_extract', '2stage'])
    parser.add_argument('--structure', '-s', type=str, required=False, help="Structural Element",
                        default='None',
                        choices=['dep', 'srl', 'const', 'None'])
    parser.add_argument('--parser', default=None, type=str, required=False, help="Parser Type")

    parser.add_argument('--prompt', '-p', type=str, default='open',
                        choices=['open', 'rel'], help="Prompt Type")
    parser.add_argument('--demo', '-d', type=str, default='random', required=False,
                        choices=['zero'],
                        help="Demonstration Retrieval Strategy")
    parser.add_argument('--model', '-m', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', required=True,
                        help="LLM")
    parser.add_argument("--reason", action='store_true', help="Add reasoning to examples")

    parser.add_argument('--data_dir', '-dir', type=str, required=True)
    parser.add_argument('--prompt_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="LLM cache directory")
    parser.add_argument('--out_path', '-out', type=str, required=True, help="Output Directory")

    parser.add_argument("--debug", action='store_true', help="When Debugging")

    args = parser.parse_args()

    if args.structure == 'None':
        args.structure = None

    try:
        main(args)
    except Exception as e:
        print(traceback.format_exc())
    print('\tDone.')
