# StructLLM

This repository contains code associated with the paper "From Syntax to Semantics: Evaluating the Impact of Linguistic Structures on LLM-Based Information Extraction" accepted at the in the Proceedings of the 1st Joint Workshop on Large Language Models and Structure Modeling, co-located with ACL 2025.

## Repository Structure
```bash
LLM4RE/
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

## Getting Started
- Model Inference: Use the main.py file in the src_jre and src_re folders to make inference for joint relation extraction and relation classification.
```bash
python ./src_re/main.py \
--task "NYT10" \
-d "knn" \
-m "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--k "5" \
--prompt "open" \
-dir <Data Directory> \
-out <Output Directory> \
--prompt_dir ./prompts

```
- Evaluation: This folder contains code to extract traditional and GenRES metrics. The CS, US, and TS metric calculation code has been derived from the [GenRES](https://github.com/pat-jj/GenRES) repository which is based this [paper](https://aclanthology.org/2024.naacl-long.155/)

