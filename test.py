
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from threading import Thread
import torch
from server.text_generation_server.prompt import sentence
import time

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
# )
model = AutoModelForCausalLM.from_pretrained(
  'meta-llama/Llama-2-7b-hf',
    # trust_remote_code=True,
    # quantization_config=nf4_config,
    # Load in bf16
    # bnb_4bit_quant_type="nf4",
    # # bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_compute_dtype=torch.float16,
    # load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map='auto'
).eval()

print('device', model.device)


tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    use_fast=True,
    # trust_remote_code=True,
)

# Run the model on the input text.
orig_input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Only choose the first 256 tokens
orig_input_ids = orig_input_ids[:, :256]

sm_input_ids = orig_input_ids.repeat(2, 1).to(model.device)

# First we warmup the model
model.generate(
    input_ids=sm_input_ids,
    do_sample=True,
    max_new_tokens=8,
    top_p=0.9,
    top_k=0,
    temperature=0.9,
    num_return_sequences=1,
)

batch_size = 1
input_ids = orig_input_ids.repeat(batch_size, 1).to(model.device)

print('done warming up...')

# Get a streamer for streaming generation
from batch_streamer import BatchTextIteratorStreamer
streamer = BatchTextIteratorStreamer(tokenizer, skip_prompt=True)

thread = Thread(target=model.generate, kwargs={
        'input_ids': input_ids,
        'do_sample': True,
        'max_new_tokens': 256,
        'top_p': 0.9,
        'top_k': 0,
        'temperature': 0.9,
        'streamer': streamer
})
thread.start()

start = time.time()
first = True
count = 0
for tokens in streamer:
    count += 1
    if first:
        print('TTFT', time.time() - start)
        first = False
        start = time.time()
    print(tokens, end='')

print()
print(f'Time to gen {count} tokens', time.time() - start)
