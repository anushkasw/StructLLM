import json
import os
from tqdm import tqdm
from utils import get_rel2prompt

class instance:
    def __init__(self, tmp_dict, rel2prompt, re_type, structure=None):
        if re_type == 'RC':
            self.id = tmp_dict["id"]
            self.sentence = " ".join(tmp_dict["token"])

            if tmp_dict["relation"] in ['no_relation', 'Other']:
                self.relation = "NONE"
            else:
                self.relation = tmp_dict["relation"]
            self.prompt_label = rel2prompt[self.relation]

            if structure and structure != 'mask':
                type_info = f'stanford_{structure}'
            else:
                type_info = f'stanford_ner'

            ss, se = tmp_dict['subj_start'], tmp_dict['subj_end']
            self.head = ' '.join(tmp_dict['token'][ss:se + 1])
            self.head_type = tmp_dict[type_info][ss]
            if self.head_type == "misc":
                self.headtype = "miscellaneous"
            elif self.head_type == 'O':
                self.headtype = "unkown"

            os, oe = tmp_dict['obj_start'], tmp_dict['obj_end']
            self.tail = ' '.join(tmp_dict['token'][os:oe + 1])
            self.tail_type = tmp_dict[type_info][os]
            if self.tail_type == "misc":
                self.tail_type = "miscellaneous"
            elif self.tail_type == 'O':
                self.tail_type = "unkown"
        else:
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

        # Mapping 'no_relation' and 'Other' labels to 'NONE'
        if args.dataset in ["semeval_nodir", "GIDS"]:
            self.rel2id['NONE'] = self.rel2id.pop('Other')
            args.na_idx = self.rel2id['NONE']
        elif args.dataset in ["tacred", "tacrev", "retacred", "dummy_tacred", "kbp37_nodir", "CORE"]:
            self.rel2id['NONE'] = self.rel2id.pop('no_relation')
            args.na_idx = self.rel2id['NONE']

        self.rel2prompt = get_rel2prompt(args.dataset, self.rel2id)

        if os.path.exists(f'{args.data_dir}/{args.dataset}/ner2id.json'):
            with open(f'{args.data_dir}/{args.dataset}/ner2id.json') as f:
                self.ner2id = json.load(f)

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
        idx_key = 'id' if self.re_type == 'RC' else 'sample_id'
        with open(example_path, "r") as f:
            for line in tqdm(f.read().splitlines()):
                tmp_dict = json.loads(line)
                examples[tmp_dict[idx_key]] = instance(tmp_dict, self.rel2prompt, self.re_type, structure=self.structure)
        return examples