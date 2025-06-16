import re


def flatten_list(labels):
    flattened = []
    for item in labels:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def get_rel2prompt(task, rel2id):
    rel2prompt = {}
    for name, id in rel2id.items():
        if task == 'wiki80':
            labels = name.split(' ')

        elif task == 'semeval_nodir':
            labels = name.split('-')

        elif task == 'FewRel':
            labels = name.split('_')

        elif task == 'FinRED':
            if name == "director_/_manager":
                labels = ['director', 'manager']
            else:
                labels = name.split('_')

        elif task == 'FIRE':
            labels = re.findall('[A-Z][^A-Z]*', name)

        elif task in ['NYT10', 'GIDS']:
            if name == 'Other':
                labels = ['None']
            elif name == '/people/person/education./education/education/institution':
                labels = ['person', 'and', 'education', 'institution']
            elif name == '/people/person/education./education/education/degree':
                labels = ['person', 'and', 'education', 'degree']
            else:
                labels = name.split('/')
                labels[-1] = "and_" + labels[-1]
                labels = labels[2:]
                for idx, lab in enumerate(labels):
                    if "_" in lab:
                        labels[idx] = lab.split("_")
                labels = flatten_list(labels)

        elif task == 'WebNLG':
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

        elif task == 'crossRE':
            if name == "win-defeat":
                labels = ['win', 'or', 'defeat']
            else:
                labels = name.split('-')

        elif task in ['tacred', 'tacrev', 'retacred', 'dummy_tacred', 'kbp37', 'tacred_new', 'retacred_new']:

            labels = [name.lower().replace("_", " ").replace("-", " ").replace("per", "person").replace("org",
                                                                                                        "organization").replace(
                "stateor", "state or ")]

        labels = [item.lower() for item in labels]

        if task == 'semeval_nodir':
            rel2prompt[name] = ' and '.join(labels).lower()
        else:
            rel2prompt[name] = ' '.join(labels).lower()
    return rel2prompt
