import json
import os
from tqdm import tqdm
from utils import get_rel2prompt

class instance:
    def __init__(self, tmp_dict, rel2prompt, re_type, structure=None):
        self.id = tmp_dict["sample_id"]
        self.sentence = tmp_dict['sentText']

        self.triples = []
        for mentions in tmp_dict['relationMentions']:
            if mentions["label"] in ['no_relation', 'Other']:
                relation = "NONE"
            else:
                relation = mentions["label"]
            prompt_label = rel2prompt[relation]

            if mentions['em1Type'] == "misc":
                headtype = "miscellaneous"
            elif mentions['em1Type'] == 'O':
                headtype = "unkown"
            else:
                headtype = mentions['em1Type']

            if mentions['em2Type'] == "misc":
                tailtype = "miscellaneous"
            elif mentions['em2Type'] == 'O':
                tailtype = "unkown"
            else:
                tailtype = mentions['em2Type']

            self.triples.append({
                'id': mentions['sent_id'] if 'sent_id' in mentions else None,
                'head': mentions['em1Text'],
                'tail': mentions['em2Text'],
                'head_type': headtype,
                'tail_type': tailtype,
                'relation': relation,
                'prompt_relation': prompt_label
            }
            )

class DataProcessor:
    def __init__(self, args, data_seed):
        self.re_type = args.re_type
        self.structure = args.structure
        self.data_dir = args.data_dir

        with open(f'{self.data_dir}/{args.dataset}/rel2id.json', "r") as f:
            self.rel2id = json.loads(f.read())

        if args.dataset in ["tacred", "tacrev", "retacred"]:
            self.rel2id['NONE'] = self.rel2id.pop('no_relation')
            args.na_idx = self.rel2id['NONE']

        self.rel2prompt = get_rel2prompt(args.dataset, self.rel2id)

        self.train_path = f'{self.data_dir}/{args.dataset}/train.jsonl'
        if data_seed:
            self.test_path = f'{self.data_dir}/{args.dataset}/test-{data_seed}.jsonl'
        else:
            self.test_path = f'{self.data_dir}/{args.dataset}/test.jsonl'

    def get_train_examples(self):
        return self.get_examples(self.train_path)

    def get_test_examples(self):
        return self.get_examples(self.test_path)

    def get_examples(self, example_path):
        examples = {}
        idx_key = 'id'
        with open(example_path, "r") as f:
            for line in tqdm(f.read().splitlines()):
                tmp_dict = json.loads(line)
                examples[tmp_dict[idx_key]] = instance(tmp_dict, self.rel2prompt, self.re_type, structure=self.structure)
        return examples