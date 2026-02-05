import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread  # Import Thread for concurrent execution

# Use GPT-2 tokenizer (standard practice for this replication)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token

# Load model from hugging face
model_id = 'SauravP97/tiny-stories-3M'
pretrained_model = AutoModelForCausalLM.from_pretrained(model_id)

# Move model to evaluation mode
pretrained_model.eval()

def stream_inference(prompt: str):
    # Prompt with a typical TinyStories opening
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