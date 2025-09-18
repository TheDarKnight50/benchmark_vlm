from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch  # <-- ADD IMPORT
import time   # <-- ADD IMPORT

class CLIPBenchmark:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading CLIP model: '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP model loaded successfully.")

    def run_zeroshot_classification(self, image_path: str, text_prompts: list):
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None, 0, 0

        inputs = self.processor(
            text=text_prompts, images=image, return_tensors="pt", padding=True
        ).to(self.device)

        # --- Start Measurement ---
        peak_memory_mb = 0.0 # Default memory to 0 for CPU
        if self.device == 'cuda': # <-- ADD THIS CHECK
            torch.cuda.reset_peak_memory_stats(self.device)
        
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # --- End Measurement ---
        end_time = time.time()
        latency = end_time - start_time
        
        if self.device == 'cuda': # <-- ADD THIS CHECK
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        
        # Calculate probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze()
        results = {prompt: prob.item() for prompt, prob in zip(text_prompts, probs)}
        
        return results, latency, peak_memory_mb