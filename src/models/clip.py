# In src/models/clip.py

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPBenchmark:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes and loads the CLIP model and processor.
        The model is loaded from the HuggingFace Hub.
        """
        print(f"Loading CLIP model: '{model_name}'...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP model loaded successfully.")

    def run_zeroshot_classification(self, image_path: str, text_prompts: list):
        """
        Performs zero-shot classification on a single image.
        
        Args:
            image_path (str): The local path to the image file.
            text_prompts (list): A list of text strings to classify the image against.
            
        Returns:
            A dictionary mapping each text prompt to its probability score.
        """
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Process the inputs
        inputs = self.processor(
            text=text_prompts, images=image, return_tensors="pt", padding=True
        )
        
        # Get model outputs
        outputs = self.model(**inputs)
        
        # Calculate probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze() # Use squeeze to get a 1D tensor
        
        # Create a dictionary of prompts to probabilities
        results = {prompt: prob.item() for prompt, prob in zip(text_prompts, probs)}
        
        return results