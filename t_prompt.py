import torch
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())


import os
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

import fire

from llama import Llama
from typing import List

generator = Llama.build(
        ckpt_dir='./llama-2-7b/',
        tokenizer_path='./llama-2-7b/tokenizer.model',
        max_seq_len=128,
        max_batch_size=4,
        )
prompts: List[str] = [
    """tranlate a sentence into a predicate:
    
    sentence: The cup is broken.
    predicate: cup_is_broken
    
    sentence: No water comes out of faucet.
    predicate: faucet_no_water
    
    sentence:  provide water for humans to drink.
    predicate: """,]



results = generator.text_completion(
    prompts,
    max_gen_len=64,
    temperature=0.6,
    top_p=0.9,
    )

for prompt, result in zip(prompts, results):
    print(prompt)
    print(f"> {result['generation']}")
    print("\n==================================\n")
