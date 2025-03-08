import gc
import json
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from timer import Timer

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class QwenModel:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.timer = Timer()
        self._load_model()

    def _load_model(self):
        """Load the model with accelerated multi-GPU device mapping."""
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            device_map="auto",  # Using automatic device map
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_responses(self, prompts):
        """Generate responses for a batch of prompts."""
        
        prompt = """
        Extract the following answers for the questions from the above conversation:
            What is your father's name?
            Which is your favorite city?
        """

        structure = """
        Give me the answer as structured JSON:
        - if there is no data return null for that key
            Example:
                {
                  "father_name": "",
                  "favorite_city": "",
                }
        """

        sys_prompt = "You are a helpful assistant. Return only the structured JSON string."

        messages_batch = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": chat + prompt + structure}
            ]
            for chat in prompts
        ]

        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

        self.timer.start()
        print("Generating responses...")

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    top_p=0.85,
                    top_k=50,
                )
            generated_responses = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            responses = self.tokenizer.batch_decode(generated_responses, skip_special_tokens=True)

            results = []
            for response in responses:
                try:
                    json_data = json.loads(response)
                    results.append(json_data)
                except Exception as e:
                    results.append({"error": str(e), "raw_response": response})

            return results

        except torch.cuda.OutOfMemoryError as e:
            raise HTTPException(status_code=500, detail=f"CUDA Out of Memory Error: {str(e)}")

        finally:
            self.timer.stop()

    def clear_gpu_memory(self):
        """Clear GPU memory."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

