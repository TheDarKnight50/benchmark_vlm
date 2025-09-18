# In src/main_evaluate.py

from models.clip import CLIPBenchmark
import os

def main():
    """
    Main function to run a simple test of the benchmarking suite.
    """
    print("--- Starting VLM Benchmark Test ---")
    
    # --- 1. Initialize the Model ---
    # This will create an instance of our CLIP class, which loads the model.
    clip_model = CLIPBenchmark()
    
    # --- 2. Prepare Data ---
    # For this test, you need a sample image.
    # TODO: Replace with your own image file.
    image_filename = "cat.jpeg" 
    
    # Check if the image exists in the project's root directory
    if not os.path.exists(image_filename):
        print(f"\nERROR: Test image '{image_filename}' not found.")
        print("Please download a sample image, name it 'cat.jpg', and place it in the 'benchmark_vlm' root directory.")
        return
        
    imagenet_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

    # --- 3. Run Inference ---
    print(f"\nRunning zero-shot classification on '{image_filename}'...")
    results = clip_model.run_zeroshot_classification(
        image_path=image_filename,
        text_prompts=imagenet_prompts
    )
    
    # --- 4. Display Results ---
    if results:
        print("\n--- Results ---")
        for prompt, score in results.items():
            print(f"{prompt}: {score:.4f}")
        print("---------------")

if __name__ == "__main__":
    main()