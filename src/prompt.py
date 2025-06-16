import json
import string


def clean(text):
    text = text.lower()
    text = text.replace('  ', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', '')
    return text


class JRE_Prompt_Processor:
    def __init__(self, args, data_processor):
        self.structure = args.structure
        self.exp = args.exp
        self.demo = args.demo
        self.prompt = args.prompt
        self.prompt_dir = args.prompt_dir
        self.parser = args.parser
        self.re_type = args.re_type
        self.data_dir = f'{args.data_dir}/{args.dataset}'
        self.reason = args.reason
        self.data_processor = data_processor

    def get_structure_info(self):
        structure_string = ""
        if self.structure == 'dep':
            structure_string = "Dependency Tree"
        elif self.structure == 'const':
            structure_string = "Constituency Tree"
        elif self.structure == 'srl':
            structure_string = "Semantic Role Labeling"
        return structure_string

    def get_structure_mapping(self):
        structure_mappings = {}
        with open(f'{self.data_dir}/mapping_{self.structure}_{self.parser}.jsonl', "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]

        for item in data:
            structure_mappings[item["sample_id"]] = item["parsed_output"]
        return structure_mappings if len(structure_mappings) > 0 else None

    def create_prompt_baseline(self, prompt_template, demo_list):
        if self.prompt.startswith('rel'):
            relations = list(self.data_processor.rel2prompt.values())
            prompt_template = prompt_template.replace("$RELATION_SET$",
                                                      '[' + ', '.join(str(x) for x in relations) + ']')
        return prompt_template

    def create_prompt_structure_extract(self, input_id, prompt_template, demo_list):
        if self.prompt.startswith('rel'):
            relations = list(self.data_processor.rel2prompt.values())
            prompt_template = prompt_template.replace("$RELATION_SET$",
                                                      '[' + ', '.join(str(x) for x in relations) + ']')

        structure_string = self.get_structure_info()
        prompt_template = prompt_template.replace("$STRUCTURE$", structure_string)

        if self.reason:
            prompt_template = prompt_template.replace("$REASON$", 'In the end please explain your resoning for '
                                                                  'extracting the triplets in less than 100 words '
                                                                  'starting with $REASON$ tag.')
        else:
            prompt_template = prompt_template.replace("$REASON$", 'Please do not add any additional information or '
                                                                  'explain how you extract them.')

        structure_mappings = self.get_structure_mapping()
        if not input_id in structure_mappings:
            structure_mappings[input_id] = ""
        prompt_template = prompt_template.replace("$STRING$", structure_mappings[input_id])
        return prompt_template

    def create_prompt(self, input, demo_list):
        with open(f'{self.prompt_dir}/{self.exp}/JRE_{self.prompt}_0.txt', 'r') as f:
                prompt_template = f.read()

        if self.exp in ['2stage']:
            prompt_template = self.create_prompt_baseline(prompt_template, demo_list)
        elif self.exp == 'structure_extract':
            prompt_template = self.create_prompt_structure_extract(input.id, prompt_template, demo_list)

        testsen = input.sentence
        prompt_template = prompt_template.replace("$TEXT$", testsen)
        messages = [{"role": "user", "content": prompt_template}]
        return messages