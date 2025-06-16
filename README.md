# StructLLM

This repository contains code associated with the paper "From Syntax to Semantics: Evaluating the Impact of Linguistic Structures on LLM-Based Information Extraction" accepted at the in the Proceedings of the 1st Joint Workshop on Large Language Models and Structure Modeling, co-located with ACL 2025.

## Repository Structure
```bash
LLM4RE/
├── Data_JRE/            # Dummy NYT10 data 
├── evaluations/         # Traditional and GenRES evaluation metric calculation
├── prompts/             # Prompts used for the pipeline
├── src/                  # Source code
└── requirements.txt     # Python dependencies
```

## Installation
To clone this repository and set up the environment:

```bash
pip install -r requirements.txt
```

## Knowledge Extraction

- Code to extract constituency and dependency trees and semantic role labelling using AllenNLP and Stanza can be found in - `extract_structure.py` 
- Use the [DeepSRL](https://github.com/luheng/deep_srl) repo to extract SRL using the `dsrl` framework. 
- The output maps should be stored within the appropriate dataset folder in `Data_JRE`.


## Inference
- Model Inference: Use the main.py file in the `src` folders to make inference for joint relation extraction.

- For baseline experiments without linguistic knowledge:
```bash
python3 src/main.py \
--re_type "JRE" \
--api_key <KEY> \
--dataset "NYT10" \
-e "2stage" \
-d "zero" \
-m "meta-llama/Meta-Llama-3.1-8B-Instruct" \
-p "open" \
-dir <BASE DIRECTORY> \
-out <OUTPUT DIRECTORY> \
--cache_dir <MODEL CACHE> \
--prompt_dir ./prompts
```

- For experiments with linguistic knowledge:
```bash
python3 src/main.py \
--re_type "JRE" \
--api_key <KEY> \
--dataset "NYT10" \
-e "structure_extract" \
-s "srl" \
-d "zero" \
-m "meta-llama/Meta-Llama-3.1-8B-Instruct" \
-p "open" \
--parser "allen" \
-dir <BASE DIRECTORY> \
-out <OUTPUT DIRECTORY> \
--cache_dir <MODEL CACHE> \
--prompt_dir ./prompts
```
- One can use the jupyter notebook `JRE_get_triples.ipynb` to postprocess the LLM output in the form of a json like dictionary with triples stored as list of

## Evaluation
- Evaluation code can be found in `./evaluation` folder.
- For TS calculation, trained LDA models are required. Use `./evaluations/create_lda_models.py` to train these models.
- TS, CS, US and traditional P,R,F1 calculations can be dome using the `cs_triples.py`, `ts_triples.py`, `us_triples.py` and `prf_triples.py` files in the evaluations folder.
```bash
python /blue/woodard/share/Relation-Extraction/LLM_feasibility/LLM_extended/evaluations/${metric}_triples.py \
    --api_key <KEY> \
    --exp "structure_extract" \
    --dataset "NYT10" \
    -m "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --base_path <BASE DIRECTORY> \
    --res_path <INPUT RESULTS FOLDER> \
    --out_path <PATH TO SAVE SCORES> \
    --embd_path <PATH TO GPT EMBEDDINGS>
```
- This code has been derived from the [GenRES](https://github.com/pat-jj/GenRES) repository which is based this [paper](https://aclanthology.org/2024.naacl-long.155/)

