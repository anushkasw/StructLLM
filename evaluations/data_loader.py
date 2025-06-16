import json


def get_RC_data(data, path):
    with open(f'{path}/Data/{data}/test.json', "r") as f:
        test_data = json.load(f)

    data_dict = {}
    for sample in test_data:
        row = {
            'text': " ".join(sample['token']).lower(),
            'relation': sample['relation']
        }
        data_dict[sample['id']] = row

    with open(f'{path}/Data/{data}/rel2id.json', "r") as f:
        rel2id = json.load(f)

    return data_dict, rel2id


def get_JRE_data(data, path):
    data_dict = {}
    with open(f'{path}/Data_JRE/{data}/test.jsonl', "r") as f:
        for line in f.read().splitlines():
            tmp_dict = json.loads(line)
            triples = []
            for triple in tmp_dict['relationMentions']:
                triples.append((triple['em1Text'].lower(), triple['label'].lower(), triple['em2Text'].lower()))
            row = {
                'text': tmp_dict['sentText'],
                'triples': triples
            }
            data_dict[tmp_dict['sample_id']] = row

    with open(f'{path}/Data_JRE/{data}/rel2id.json', "r") as f:
        rel2id = json.load(f)

    if data == 'FewRel':
        rel2id['original_language_of_film_or_tv_show'] = rel2id.pop('original_language_of_film_or_TV_show')
    return data_dict, rel2id