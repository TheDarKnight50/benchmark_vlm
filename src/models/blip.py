from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch  # <-- ADD IMPORT
import time   # <-- ADD IMPORT

class BLIPBenchmark:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        print(f"Loading BLIP model: '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("BLIP model loaded successfully.")

    def run_image_captioning(self, image_path: str):
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None, 0, 0

        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        
        # --- Start Measurement ---
        peak_memory_mb = 0.0 # Default memory to 0 for CPU
        if self.device == 'cuda': # <-- ADD THIS CHECK
            torch.cuda.reset_peak_memory_stats(self.device)
            
        start_time = time.time()
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)
            
        # --- End Measurement ---
        end_time = time.time()
        latency = end_time - start_time

        if self.device == 'cuda': # <-- ADD THIS CHECK
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption, latency, peak_memory_mb