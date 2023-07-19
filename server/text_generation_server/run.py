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

model_id = 'meta-llama/Llama-2-70b-hf'
path_to_model=''
shared = True
quantize = None
dtype = torch.float16

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
        device_id=model.device_id,
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
            num_tokens += 1
            if first:
                first = False
                print('Time to first token!', time.time() - start)
                start = time.time()
            else:
                self.cache.set(batch)

            generations, batch = self.model.generate_token(batch)
            yield generations

        print(f'Time to gen {num_tokens} tokens!', time.time() - start)


    def warmup(self, sentence):
        batch = self.full_get_batch(sentence)
        self.model.warmup(batch, max_total_tokens=1000)


def main(num_batches = 1, prompt_len=512, gen_size=64):
    model = FlashLlama(
        model_id=model_id,
        quantize=quantize,
        dtype=dtype,
        trust_remote_code=True,
    )

    cache = Cache()

    generator = FlashLlamaGenerator(
        model=model,
        cache=cache,
        num_batches=num_batches,
        prompt_len=prompt_len,
        gen_size=gen_size,
    )

    print('Warming up...')
    generator.warmup(sentence)

    print('Decoding...')
    for generations in generator.decode(sentence):
        first = generations[0]
        print(first.generated_text, end='')


main()