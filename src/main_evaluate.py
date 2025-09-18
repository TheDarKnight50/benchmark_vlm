from models.clip import CLIPBenchmark
from models.blip import BLIPBenchmark
import os

def main():
    print("--- Starting VLM Benchmark Test ---")
    
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
    # Unpack the new return values
    clip_results, clip_latency, clip_memory = clip_model.run_zeroshot_classification(
        image_path=image_filename,
        text_prompts=imagenet_prompts
    )
    
    if clip_results:
        print("\n--- CLIP Results ---")
        for prompt, score in clip_results.items():
            print(f"{prompt}: {score:.4f}")
        # Print the new metrics
        print(f"Inference Latency: {clip_latency:.4f} seconds")
        print(f"Peak GPU Memory: {clip_memory:.2f} MB")
        print("--------------------")

    # =======================================================
    #               BLIP Captioning Test
    # =======================================================
    blip_model = BLIPBenchmark()

    print(f"\nRunning BLIP image captioning on '{image_filename}'...")
    # Unpack the new return values
    caption, blip_latency, blip_memory = blip_model.run_image_captioning(image_path=image_filename)

    if caption:
        print("\n--- BLIP Results ---")
        print(f"Generated Caption: {caption}")
        # Print the new metrics
        print(f"Inference Latency: {blip_latency:.4f} seconds")
        print(f"Peak GPU Memory: {blip_memory:.2f} MB")
        print("--------------------")

if __name__ == "__main__":
    main()