from time import sleep
import pickle
from openai import OpenAI
import numpy as np
import os


class EMBD_processor():
    def __init__(self, args):
        self.embd_path = args.embd_path
        with open(self.embd_path,
                  'rb') as handle:
            self.ELE_EMB_DICT = pickle.load(handle)
        self.client = OpenAI(api_key=args.api_key)

    def embedding_retriever(self, term):
        # print('get embds')
        i = 0
        while (i < 100):
            try:
                # Send the request and retrieve the response
                response = self.client.embeddings.create(
                    input=str(term),
                    model="text-embedding-ada-002"
                )

                # Extract the text embeddings from the response JSON
                embedding = response.data[0].embedding

                return embedding

            except Exception as e:
                i += 1
                print(f"Error in gpt_instruct: \"{term}\". Retrying...")
                sleep(5)

        return np.zeros(1536).tolist()

    def ele_proxy(self, element):
        return np.zeros(1536).tolist()

    def check_triple(self, element):
        if element not in self.ELE_EMB_DICT:
            self.ELE_EMB_DICT[element] = self.embedding_retriever(element)

    def get_gt_embds(self, relations, check=True):
        gt_relation_emb_store = {}
        for rel in relations:
            if check:
                self.check_triple(rel)
            gt_relation_emb_store[rel] = self.ELE_EMB_DICT[rel]
        return gt_relation_emb_store

    def get_ent_embds(self, ent, check=True):
        if check:
            self.check_triple(ent)
        return self.ELE_EMB_DICT[ent]

    def get_triple_embds(self, gt_triple_list, check=True):
        gt_triple_emb_store = {}
        gt_relation_emb_store = {}
        for triple in gt_triple_list:
            triple_str = str(triple)
            if check:
                self.check_triple(triple[0])
                self.check_triple(triple[1])
                self.check_triple(triple[2])

            entity_emb = np.add(self.ELE_EMB_DICT[triple[0]], self.ELE_EMB_DICT[triple[2]])
            triple_emb = np.add(np.array(entity_emb), np.array(self.ELE_EMB_DICT[triple[1]]))
            # emb_ = np.concatenate([genres.ele_emb_dict[triple[0]], genres.ele_emb_dict[triple[1]]])
            # triple_emb = np.concatenate([emb_, genres.ele_emb_dict[triple[2]]])
            gt_triple_emb_store[triple_str] = triple_emb.tolist()
            gt_relation_emb_store[triple_str] = self.ELE_EMB_DICT[triple[1]]
        return gt_triple_emb_store, gt_relation_emb_store

    def get_per_triple_embds(self, triple, check=True):
        if check:
            self.check_triple(triple[0])
            self.check_triple(triple[1])
            self.check_triple(triple[2])

        entity_emb = np.add(self.ELE_EMB_DICT[triple[0]], self.ELE_EMB_DICT[triple[2]])
        triple_emb = np.add(np.array(entity_emb), np.array(self.ELE_EMB_DICT[triple[1]]))
        return triple_emb

    def save_embeddings(self):
        os.remove(self.embd_path)
        with open(self.embd_path, 'wb') as handle:
            pickle.dump(self.ELE_EMB_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)
