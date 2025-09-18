# In src/main_evaluate.py

from models.clip import CLIPBenchmark
from models.blip import BLIPBenchmark  # <-- ADD THIS IMPORT
import os

def main():
    """
    Main function to run a simple test of the benchmarking suite.
    """
    print("--- Starting VLM Benchmark Test ---")
    
    # --- Check for test image ---
    image_filename = "cat.jpeg" 
    if not os.path.exists(image_filename):
        print(f"\nERROR: Test image '{image_filename}' not found.")
        print(f"Please place a sample image named '{image_filename}' in the root directory.")
        return
        
    # =======================================================
    #               CLIP Zero-Shot Test
    # =======================================================
    clip_model = CLIPBenchmark()
    imagenet_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

    print(f"\nRunning CLIP zero-shot classification on '{image_filename}'...")
    clip_results = clip_model.run_zeroshot_classification(
        image_path=image_filename,
        text_prompts=imagenet_prompts
    )
    
    if clip_results:
        print("\n--- CLIP Results ---")
        for prompt, score in clip_results.items():
            print(f"{prompt}: {score:.4f}")
        print("--------------------")

    # =======================================================
    #               BLIP Captioning Test
    # =======================================================
    blip_model = BLIPBenchmark() # <-- INSTANTIATE BLIP MODEL

    print(f"\nRunning BLIP image captioning on '{image_filename}'...")
    caption = blip_model.run_image_captioning(image_path=image_filename) # <-- RUN CAPTIONING

    if caption:
        print("\n--- BLIP Results ---")
        print(f"Generated Caption: {caption}") # <-- PRINT THE CAPTION
        print("--------------------")


if __name__ == "__main__":
    main()