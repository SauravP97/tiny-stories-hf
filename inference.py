from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from transformers import logging

import sys
import argparse
from colorama import Fore, init

def stream_inference(prompt: str):
    # Use GPT-2 tokenizer (standard practice for this replication)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model from hugging face
    model_id = 'SauravP97/tiny-stories-3M'
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_id)

    # Move model to evaluation mode
    pretrained_model.eval()

    # Prompt with a typical TinyStories opening
    inputs = tokenizer(prompt, return_tensors="pt").to(pretrained_model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=False)

    # Generate
    return pretrained_model.generate(
        inputs.input_ids, 
        max_new_tokens=150, 
        do_sample=True,
        temperature=0.7, 
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

init(autoreset=True)
logging.disable_progress_bar()
logging.set_verbosity_error()

parser = argparse.ArgumentParser(description="Executing Deep Researcher")
parser.add_argument(
    "prompt",
    help="The prompt from the user for the SLM",
)
args = parser.parse_args()

sys.stdout.write(Fore.GREEN)
stream_inference(args.prompt)