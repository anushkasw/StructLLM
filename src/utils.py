import json
import os
import re


def flatten_list(labels):
    flattened = []
    for item in labels:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def check_output_validity(filename, test_dict):
    incomplete_flag = False
    test_completed = None
    if os.path.exists(filename):
        with open(filename) as f:
            batch = f.read().splitlines()
        test_completed = {json.loads(line)['id']: json.loads(line) for line in batch if line != ""}
        if len(test_completed) != len(test_dict):
            print(f'\tSome results already processed. Setting incomplete_flag to True')
            incomplete_flag = True
    else:
        incomplete_flag = False
    return incomplete_flag, test_completed


def get_rel2prompt(dataset, rel2id):
    rel2prompt = {}
    for name, id in rel2id.items():
        if dataset in ['NYT10', 'GIDS', 'NYT10_new']:
            if name == 'Other':
                labels = ['None']
            elif name == '/people/person/education./education/education/institution':
                labels = ['person', 'and', 'education', 'institution']
            elif name == '/people/person/education./education/education/degree':
                labels = ['person', 'and', 'education', 'degree']
            else:
                labels = name.split('/')
                labels[-1] = "and_"+labels[-1]
                labels = labels[2:]
                for idx, lab in enumerate(labels):
                    if "_" in lab:
                        labels[idx] = lab.split("_")
                labels = flatten_list(labels)

        elif dataset == 'crossRE':
            if name == "win-defeat":
                labels = ['win', 'or', 'defeat']
            else:
                labels = name.split('-')

        elif dataset in ['tacred', 'tacrev', 'retacred']:
            labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                        "organization").replace(
                "stateor", "state or ")]

        labels = [item.lower() for item in labels]
        rel2prompt[name] = ' '.join(labels).upper()
    return rel2prompt