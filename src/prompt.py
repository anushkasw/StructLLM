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

    def get_demo(self, demo_list, structure_mappings=None):
        demo_prompt = ''
        for demo in demo_list:
            sentence = demo.sentence

            entities = set()
            for triple in demo.triples:
                entities.add(triple['head'].upper())
                entities.add(triple['tail'].upper())
            ent_str = "["
            for ent in entities:
                ent_str += f'{ent}, '
            ent_str = ent_str[:-2]
            ent_str += "]"

            triple_str = "["
            for triple in demo.triples:
                subj = triple['head'].upper()
                obj = triple['tail'].upper()
                relation = triple['prompt_relation'].upper()
                triple_str += f'[{subj}, {relation}, {obj}]'
            triple_str += "]"

            if self.structure:
                structure_string = self.get_structure_info()
                demo_prompt += f"Context: {sentence}\n{structure_string}:{structure_mappings[demo.id]}\nGiven the context, the list of entities are {ent_str} and the list of triplets are {triple_str}"
            else:
                demo_prompt += f"Context: {sentence}\nGiven the context, the list of entities are {ent_str} and the list of triplets are {triple_str}"

            # if self.reason:
            #     demo_prompt += f"Reason:\n"
            #     for triple in demo.triples:
            #         demo_prompt += f"{clean(reason[triple['id']])}\n"
        return demo_prompt

    def create_prompt_baseline(self, prompt_template, demo_list):
        if self.prompt.startswith('entrel'):
            relations = list(self.data_processor.rel2prompt.values())
            entities = list(self.data_processor.ner2id.keys())
            prompt_template = prompt_template.replace("$RELATION_SET$",
                                                      '[' + ', '.join(str(x) for x in relations) + ']')
            prompt_template = prompt_template.replace("$ENTITY_SET$", '[' + ', '.join(str(x) for x in entities) + ']')
        elif self.prompt.startswith('rel'):
            relations = list(self.data_processor.rel2prompt.values())
            prompt_template = prompt_template.replace("$RELATION_SET$",
                                                      '[' + ', '.join(str(x) for x in relations) + ']')

        if not self.demo == 'zero':
            examples = self.get_demo(demo_list, self.data_processor.reasons)
            prompt_template = prompt_template.replace("$EXAMPLES$", examples)
        return prompt_template

    def create_prompt_structure_extract(self, input_id, prompt_template, demo_list):
        if self.prompt.startswith('entrel'):
            pass
        elif self.prompt.startswith('rel'):
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

        if not self.demo == 'zero':
            examples = self.get_demo(demo_list, structure_mappings=structure_mappings)
            prompt_template = prompt_template.replace("$EXAMPLES$", examples)
        return prompt_template

    def create_prompt(self, input, demo_list):
        if not self.demo == 'zero':
            with open(f'{self.prompt_dir}/{self.exp}/{self.re_type}_{self.prompt}.txt', 'r') as f:
                prompt_template = f.read()
        else:
            with open(f'{self.prompt_dir}/{self.exp}/JRE_{self.prompt}_0.txt', 'r') as f:
                prompt_template = f.read()

        if self.exp in ['2stage', '1stage']:
            prompt_template = self.create_prompt_baseline(prompt_template, demo_list)
        elif self.exp == 'structure_extract':
            prompt_template = self.create_prompt_structure_extract(input.id, prompt_template, demo_list)

        testsen = input.sentence
        prompt_template = prompt_template.replace("$TEXT$", testsen)
        messages = [{"role": "user", "content": prompt_template}]
        return messages


class RC_Prompt_Processor:
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
        self.relations = list(self.data_processor.rel2prompt.values())

        self.choice_dict = {}
        if self.exp == 'multi-choice':
            choice_template = ""
            for cnt, relation in enumerate(self.relations):
                self.choice_dict[relation] = cnt+1
                choice_template += f"[{cnt+1}]. {relation}\n"
            self.choice_template = choice_template[:-1]

    def get_demo(self, demo_list, prompt_type):
        demo_prompt = ''
        for demo in demo_list:
            sentence = demo.sentence
            subj = demo.head
            obj = demo.tail
            subj_type = demo.head_type
            obj_type = demo.tail_type
            relation = demo.prompt_label

            if prompt_type == 'ent' or prompt_type == 'entrel':
                demo_prompt += (
                    f"Context: {sentence}\nGiven the context, find relation between {subj} of type {subj_type} "
                    f"and {obj} of type {obj_type} from the following options:\n{self.choice_template}\nThe correct choice: {self.choice_dict[relation]}\n\n")
            else:
                demo_prompt += (
                    f"Context: {sentence}\nGiven the context, find relation between {subj} and {obj} from the following options:\n{self.choice_template}\nThe correct choice: {self.choice_dict[relation]}\n\n")
        return demo_prompt

    def create_prompt(self, input, demo_list):
        with open(f'{self.prompt_dir}/{self.exp}/{self.re_type}_{self.prompt}.txt', 'r') as f:
            prompt_template = f.read()

        if self.prompt == 'rel' or self.prompt == 'entrel':
            prompt_template = prompt_template.replace("$RELATION_SET$",
                                                      '[' + ', '.join(str(x) for x in self.relations) + ']')

        if self.prompt == 'ent':
            type_dict = {'pos': 'part-of-speech', 'ner': 'NER', 'deprel': 'dependency'}
            prompt_template = prompt_template.replace("$TYPE$", type_dict[self.structure])

        if self.exp == 'multi-choice':
            prompt_template = prompt_template.replace("$CHOICES$", self.choice_template)

        if not self.demo == 'zero':
            examples = self.get_demo(demo_list, self.prompt)
            prompt_template = prompt_template.replace("$EXAMPLES$", examples)
        else:
            prompt_template = prompt_template.replace("$EXAMPLES$", "")

        testsen = input.sentence
        subj = input.head
        obj = input.tail
        subj_type = input.head_type
        obj_type = input.tail_type

        prompt_template = prompt_template.replace("$TEXT$", testsen)
        prompt_template = prompt_template.replace("$SUBJECT$", subj)
        prompt_template = prompt_template.replace("$OBJECT$", obj)

        if self.prompt == 'ent' or self.prompt == 'entrel':
            prompt_template = prompt_template.replace("$SUBJ_TYPE$", subj_type)
            prompt_template = prompt_template.replace("$OBJ_TYPE$", obj_type)
        messages = [{"role": "system", "content": 'You are a knowledgeable person. You will solve the relation extraction task.'},
                    {"role": "user", "content": prompt_template}]
        return messages
