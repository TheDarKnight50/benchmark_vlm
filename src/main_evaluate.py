# In src/main_evaluate.py

from models.clip import CLIPBenchmark
from models.blip import BLIPBenchmark
import os
import glob
import csv

def main():
    """
    Main function to run the full benchmarking suite over a dataset.
    """
    print("--- Starting VLM Benchmark ---")

    # --- 1. Configuration ---
    # Define the directory of images to process and where to save results.
    DATASET_DIR = "data/sample_images"
    OUTPUT_CSV = "results/leaderboard.csv"
    
    # Use glob to find all image files in the directory
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(DATASET_DIR, ext)))

    if not image_paths:
        print(f"Error: No images found in '{DATASET_DIR}'. Please add some images to process.")
        return

    print(f"Found {len(image_paths)} images to benchmark.")

    # --- 2. Initialize Models ---
    # Load all models once at the beginning to be efficient.
    print("\nLoading models...")
    clip_model = CLIPBenchmark()
    blip_model = BLIPBenchmark()
    print("All models loaded.")

    # --- 3. Run Benchmark Loop ---
    # This list will store a dictionary for each result.
    all_results = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        print(f"\n--- Processing: {image_name} ---")

        # --- Run CLIP ---
        clip_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a car"]
        clip_results, clip_latency, clip_memory = clip_model.run_zeroshot_classification(
            image_path=image_path,
            text_prompts=clip_prompts
        )
        if clip_results:
            # Find the prompt with the highest score for a concise result
            top_prediction = max(clip_results, key=clip_results.get)
            all_results.append({
                'model': 'CLIP',
                'image': image_name,
                'task': 'Zero-Shot Classification',
                'result': f"Top Prediction: {top_prediction} ({clip_results[top_prediction]:.2f})",
                'latency_s': f"{clip_latency:.4f}",
                'memory_mb': f"{clip_memory:.2f}"
            })

        # --- Run BLIP ---
        caption, blip_latency, blip_memory = blip_model.run_image_captioning(
            image_path=image_path
        )
        if caption:
            all_results.append({
                'model': 'BLIP',
                'image': image_name,
                'task': 'Image Captioning',
                'result': caption,
                'latency_s': f"{blip_latency:.4f}",
                'memory_mb': f"{blip_memory:.2f}"
            })

    # --- 4. Save Results to CSV ---
    if not all_results:
        print("No results were generated.")
        return
        
    # Get the headers from the keys of the first dictionary in the list.
    headers = all_results[0].keys()
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n--- Benchmark Complete ---")
    print(f"Results for {len(image_paths)} images saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()