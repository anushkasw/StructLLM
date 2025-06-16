import argparse
import os
import json

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from pathlib import Path

from gensim.matutils import kullback_leibler
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

from embeddings import EMBD_processor
from rel_verbaliser import get_rel2prompt
from data_loader import get_RC_data, get_JRE_data


class TS_calculator:
    def __init__(self, dictionary, lda_model, embedder):
        self.dictionary = dictionary
        self.lda_model = lda_model
        self.embedder = embedder

    def preprocess(self, text):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        return [word for word in words if word not in stop_words and word.isalnum()]

    def get_ts_scores(self, tmp_dict, data_dict):
        all_ts_scores = {}

        ts_dict = {}
        for dict_ in tmp_dict:
            source_text = data_dict[dict_['id']]['text']
            triples = dict_['pred_label']
            triples_str = ''
            if triples and triples != 'Other':
                for triple in triples:
                    if len(triple) == 3:
                        triples_str += f"{triple[0]} {triple[1]} {triple[2]} ."
                    else:
                        continue
            else:
                continue
            processed_source = self.preprocess(source_text)
            processed_triples = self.preprocess(triples_str)
            source_corpus = self.dictionary.doc2bow(processed_source)
            triples_corpus = self.dictionary.doc2bow(processed_triples)

            source_dist = self.lda_model.get_document_topics(source_corpus, minimum_probability=0)
            triples_dist = self.lda_model.get_document_topics(triples_corpus, minimum_probability=0)

            ts_score = math.exp(-kullback_leibler(source_dist, triples_dist))
            # all_ts_scores[source_text] = ts_score
            ts_dict[dict_['id']] = ts_score

        # average_ts_score = sum(all_ts_scores.values()) / len(all_ts_scores)
        return ts_dict


def main(args):
    embdder = EMBD_processor(args)

    data = args.dataset
    data_dict, rel2id = get_JRE_data(data, args.base_path)
    dictionary = joblib.load(
        open(f'{args.base_path}/topical_models/{data}/dictionary.joblib', 'rb'))
    lda_model = joblib.load(
        open(f'{args.base_path}/topical_models/{data}/lda.joblib', 'rb'))
    ts_calculator = TS_calculator(dictionary, lda_model, embdder)

    model = args.model

    # if args.exp=='plm':
    #     models = ["PLM/UniRel", "PLM/RIFRE", "PLM/SPN4RE", "PLM/TDEER"]
    # else:
    #     models = ["OpenAI/gpt-4o", "openchat/openchat_3.5", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #               "mistralai/Mistral-Nemo-Instruct-2407", "google/gemma-2-9b-it"]

    # for model in models:
    files = list(
        Path(f'{args.res_path}/JRE/{args.exp}/{data}/{model}'
                ).rglob('*.jsonl'))

    for file in tqdm(files):
        if not file.name.startswith(('cs', 'us', 'ts', 'prf', 'fs')):
            outfile = file.with_name(f'ts_{file.name}')
            if not os.path.exists(outfile) or args.overwrite:
                res_dict = []
                with open(file, "r") as f:
                    for line in f.read().splitlines():
                        sample = json.loads(line)
                        res_dict.append(sample)
                ts_dict = ts_calculator.get_ts_scores(res_dict, data_dict)

                with open(outfile, "w") as f:
                    json.dump(ts_dict, f)


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
