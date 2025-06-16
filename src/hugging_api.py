import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_init(model_name, cache_dir, device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = model.eval()
    return tokenizer, model


def model_inference(tokenizer, model, prompt, max_new_tokens, device='cuda'):
    tokenized_chat = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")
    tokenized_chat = tokenized_chat.to(device)
    outputs = model.generate(tokenized_chat, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id
                                   )
    # generated_ids = outputs.sequences
    # scores = outputs.scores
    decoded = tokenizer.batch_decode(outputs)
    return decoded[0]
