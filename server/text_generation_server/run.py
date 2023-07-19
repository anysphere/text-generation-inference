import random
import time
import numpy as np
from text_generation_server.models import Model, get_model
import math
import torch
from typing import List
from text_generation_server.models.flash_llama import (
    FlashLlama,
)
from text_generation_server.models.flash_causal_lm import FlashCausalLMBatch
from text_generation_server.utils import StoppingCriteria, HeterogeneousNextTokenChooser
from transformers import PreTrainedTokenizerBase
from text_generation_server.cache import Cache
from text_generation_server.prompt import sentence
import sys
import os

# redirect stdout to nothing if not rank 0


model_id = 'meta-llama/Llama-2-70b-hf'
model_id = 'meta-llama/Llama-2-7b-hf'
path_to_model=''
shared = True
quantize = None
dtype = torch.bfloat16

BLOCK_SIZE = 16

def get_batch_helper(sentence: str, num_batches: int, prompt_len: int, gen_size: int, device_id: str, tokenizer: PreTrainedTokenizerBase):
    return FlashCausalLMBatch.from_sentences(
         sentences=[sentence for _ in range(num_batches)],
         tokenizer=tokenizer,
         dtype=dtype,
         device=device_id,
         max_truncation=prompt_len,
         max_new_tokens=gen_size,
    )

def get_batch(
    sentence: str,
    model: FlashLlama,
    num_batches: int,
    prompt_len: int,
    gen_size: int,
):
    return get_batch_helper(
        sentence=sentence,
        num_batches=num_batches,
        prompt_len=prompt_len,
        gen_size=gen_size,
        device_id=model.device,
        tokenizer=model.tokenizer,
    )


class FlashLlamaGenerator:
    def __init__(self, model: FlashLlama, cache: Cache, num_batches: int, prompt_len: int, gen_size: int):
        self.model = model
        self.cache = cache
        self.num_batches = num_batches
        self.prompt_len = prompt_len
        self.gen_size = gen_size

    def full_get_batch(self, sentence: str):
        return get_batch(
            sentence=sentence,
            model=self.model,
            num_batches=self.num_batches,
            prompt_len=self.prompt_len,
            gen_size=self.gen_size,
        )

    def prefill(self, sentence: List[int]):

        batch = self.full_get_batch(sentence)
        generations, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)
        return generations, next_batch

    def decode(self, sentence: List[int]):
        batch = self.full_get_batch(sentence)
        first = True

        start = time.time()
        num_tokens = 0
        while batch:
            generations, batch = self.model.generate_token(batch)

            num_tokens += 1
            if first:
                first = False
                print('Time to first token!', time.time() - start)
                start = time.time()

            self.cache.set(batch)
            yield generations

        print(f'Time to gen {num_tokens} tokens!', time.time() - start)


    def warmup(self, sentence):
        batch = self.full_get_batch(sentence)
        self.model.warmup(batch, max_total_tokens=1024)


def run_experiment(model: FlashLlama, cache: Cache, num_batches = 1, prompt_len=512, gen_size=64, first=False):

    generator = FlashLlamaGenerator(
        model=model,
        cache=cache,
        num_batches=num_batches,
        prompt_len=prompt_len,
        gen_size=gen_size,
    )

    print('Warming up...')
    if first:
        for _ in range(3):
            generator.warmup(sentence)

    print('Decoding...')
    for generations in generator.decode(sentence):
        if len(generations) != 0:
            first = generations[0]
            if first.generated_text != None:
                print(first.generated_text.text)

    print()


def main():
    model = FlashLlama(
        model_id=model_id,
        quantize=quantize,
        dtype=dtype,
        trust_remote_code=True,
    )
    if torch.distributed.get_rank() != 0:
        sys.stdout = open(os.devnull, "w")

    cache = Cache()

    while True:
        if (torch.distributed.get_rank() == 0):
            prompt_len = int(input('prompt_len: '))
            gen_size = int(input('gen_size: '))
            data = torch.tensor([prompt_len, gen_size])
        else:
            data = torch.zeros(2)

        torch.distributed.broadcast(data, 0)

        prompt_len = int(data[0])
        gen_size = int(data[1])

        for i, batch_size in enumerate([1, 2, 4, 8]):
            print(f'batch_size: {batch_size}')
            run_experiment(
                model=model,
                cache=cache,
                num_batches=batch_size,
                prompt_len=prompt_len,
                gen_size=gen_size,
                first = i == 0,
            )

main()