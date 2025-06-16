import os
import argparse
from allennlp.predictors.predictor import Predictor
import json
import stanza

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency,mwt,lemma,depparse')

# Load pre-trained models
dep_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
constituency_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
srl_predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")


def get_dependency_parse(parser, sentence):
    """Perform dependency parsing and return an edge list format"""
    if parser == "allen":
        result = dep_predictor.predict(sentence)
        words = result["words"]
        heads = result["predicted_heads"]
        dependencies = result["predicted_dependencies"]
    else:
        doc = nlp(sentence)
        words, heads, dependencies = [], [], []
        for sent in doc.sentences:
            for word in sent.words:
                words.append(word.text)
                heads.append(word.head)
                dependencies.append(word.deprel)
    edges = [
        (words[heads[i] - 1] if heads[i] > 0 else "ROOT", dependencies[i], words[i])
        for i in range(len(words))
    ]
    return "\n".join([f"({head}, {rel}, {dep})" for head, rel, dep in edges])


def get_constituency_parse(parser, sentence):
    """Perform constituency parsing and return a tree representation"""
    if parser == "allen":
        result = constituency_predictor.predict(sentence)
        return result["trees"]
    else:
        doc = nlp(sentence)
        parser = []
        for sentence in doc.sentences:
            parser.append(f"{sentence.constituency}")
        return "\n".join(parser)


def get_srl(parser, sentence):
    """Perform Semantic Role Labeling (SRL)"""
    if parser == "allen":
        result = srl_predictor.predict(sentence)
        srl_outputs = []

        for verb in result["verbs"]:
            srl_outputs.append(f"Predicate: {verb['verb']}\nSRL: {verb['description']}")

    return "\n\n".join(srl_outputs)


def main(args):
    if args.structure in ["dep", "const"]:
        parsers = ['stanza', 'allen']
    else:
        parsers = ['allen']

    for parser in parsers:
        print(f"Parsing using {parser} parser")
        out_file = f'{args.data_dir}/{args.dataset}/mapping_{args.structure}_{parser}.jsonl'
        if not os.path.exists(out_file):
            examples = {}
            for data_type in ['test']:
                with open(f'{args.data_dir}/{args.dataset}/{data_type}.jsonl', "r") as f:
                    for line in f:
                        res_dict = json.loads(line.strip())
                        try:
                            if args.structure == 'dep':
                                examples[res_dict['sample_id']] = get_dependency_parse(parser, res_dict["sentText"])
                            elif args.structure == 'srl':
                                examples[res_dict['sample_id']] = get_srl(parser, res_dict["sentText"])
                            elif args.structure == 'const':
                                examples[res_dict['sample_id']] = get_constituency_parse(parser, res_dict["sentText"])
                        except:
                            continue

                with open(out_file, "w") as f_out:
                    for sample_id, parsed_output in examples.items():
                        json.dump({"sample_id": sample_id, "parsed_output": parsed_output}, f_out)
                        f_out.write("\n")
        else:
            print(f"Parser mapping for {parser} already exists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StructureModelling')
    parser.add_argument("--use_cuda", action='store_true',
                        help="if GPUs available")
    # Required
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--structure', '-s', type=str, required=True)

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
    print('\tDone.')
