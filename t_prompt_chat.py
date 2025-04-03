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
        ckpt_dir='./llama-2-7b-chat/',
        tokenizer_path='./llama-2-7b-chat/tokenizer.model',
        max_seq_len=512,
        max_batch_size=6,
        )

#dialogs: List[str] = [
#    [{"role": "user", "content": "is it suitable that a robot cleans a table using a vacuum in a dining room if there is a power outage? Please answer yes or no."}],
#]

dialogs = [
        [   {"role": "system", "content": "Please answer yes or no"},
            {"role": "user", "content": "is it suitable that a robot cleans a table using a vacuum in a dining room if there is a power outage? "}],






    ]

results = generator.chat_completion(
        dialogs
    )

for dialog, result in zip(dialogs, results):
    for msg in dialog:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print(
        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    )
    print("\n==================================\n")
