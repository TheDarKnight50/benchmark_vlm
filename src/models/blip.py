# In src/models/blip.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

class BLIPBenchmark:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        """
        Initializes and loads the BLIP model and processor for image captioning.
        """
        print(f"Loading BLIP model: '{model_name}'...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        print("BLIP model loaded successfully.")

    def run_image_captioning(self, image_path: str):
        """
        Generates a caption for a single image.
        
        Args:
            image_path (str): The local path to the image file.
            
        Returns:
            A string containing the generated caption.
        """
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Process the image
        inputs = self.processor(raw_image, return_tensors="pt")
        
        # Generate a caption (token IDs)
        out = self.model.generate(**inputs, max_new_tokens=50)
        
        # Decode the token IDs to a human-readable string
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return caption