import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread  # Import Thread for concurrent execution

# Use GPT-2 tokenizer (standard practice for this replication)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load model from hugging face
model_id_3m = 'SauravP97/tiny-stories-3M'
model_id_19m = 'SauravP97/tiny-stories-19M'
pretrained_model_3M = AutoModelForCausalLM.from_pretrained(model_id_3m)
pretrained_model_19M = AutoModelForCausalLM.from_pretrained(model_id_19m)

# Move model to evaluation mode
pretrained_model_3M.eval()
pretrained_model_19M.eval()

def stream_inference(prompt: str, model_size: str):
    if model_size == "3M":
        logging.info("Using 3M model for inference")
        pretrained_model = pretrained_model_3M
    elif model_size == "19M":
        logging.info("Using 19M model for inference")
        pretrained_model = pretrained_model_19M
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(pretrained_model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    model_generate_args = {
        "inputs": inputs.input_ids,
        "max_new_tokens": 150, 
        "do_sample": True,
        "temperature": 0.7, 
        "top_k": 50,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }
    thread = Thread(
        target=pretrained_model.generate,
        kwargs=model_generate_args,
    )
    thread.start()

    for text_token in streamer:
        time.sleep(0.01)  # Simulate real-time output with a short delay
        yield text_token  # Yield the accumulated text

    thread.join()