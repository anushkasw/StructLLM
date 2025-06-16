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
        if dataset == 'wiki80':
            labels = name.split(' ')

        elif dataset == 'semeval_nodir':
            labels = name.split('-')

        elif dataset == 'FewRel':
            labels = name.split('_')

        elif dataset == 'CORE':
            labels = name.split('_')

        elif dataset in ['NYT10', 'GIDS', 'NYT10_new']:
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

        elif dataset == 'FinRED':
            if name == "director_/_manager":
                labels = ['director', 'manager']
            else:
                labels = name.split('_')

        elif dataset == 'FIRE':
            labels = re.findall('[A-Z][^A-Z]*', name)

        elif dataset == 'WebNLG':
            name_mod = re.sub(r"['()]", '', name)
            labels = name_mod.split(' ')

            if len(labels) == 1:
                label0 = labels[0]
                if "_" in label0:
                    labels = label0.split("_")

                    for idx, lab in enumerate(labels):
                        if any(char.isupper() for char in lab) and not lab.isupper():
                            l = re.split(r'(?=[A-Z])', lab)
                            if l[0] == "":
                                l = l[1:]
                            labels[idx] = l

                    labels = flatten_list(labels)

                elif any(char.isupper() for char in label0):
                    labels = re.split(r'(?=[A-Z])', label0)

        elif dataset == 'crossRE':
            if name == "win-defeat":
                labels = ['win', 'or', 'defeat']
            else:
                labels = name.split('-')

        elif dataset in ['tacred', 'tacrev', 'retacred', 'dummy_tacred', 'kbp37', 'tacred_new', 'retacred_new']:
            labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                        "organization").replace(
                "stateor", "state or ")]

        labels = [item.lower() for item in labels]

        if dataset == 'semeval_nodir':
            rel2prompt[name] = ' and '.join(labels).upper()
        else:
            rel2prompt[name] = ' '.join(labels).upper()
    return rel2prompt