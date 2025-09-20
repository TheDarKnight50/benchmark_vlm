# In src/main_evaluate.py

from models.clip import CLIPBenchmark
from models.blip import BLIPBenchmark
from models.blip_vqa import BLIPVQABenchmark # <-- CHANGE THIS IMPORT
import os
import glob
import csv

def main():
    """
    Main function to run the full benchmarking suite over a dataset.
    """
    print("--- Starting VLM Benchmark ---")

    # --- 1. Configuration ---
    DATASET_DIR = "data/sample_images"
    OUTPUT_CSV = "results/leaderboard.csv"
    
    image_paths = glob.glob(os.path.join(DATASET_DIR, '*.[jp][pn][jpe]g'))

    if not image_paths:
        print(f"Error: No images found in '{DATASET_DIR}'.")
        return
    print(f"Found {len(image_paths)} images to benchmark.")

    # --- 2. Initialize Models ---
    print("\nLoading models...")
    clip_model = CLIPBenchmark()
    blip_caption_model = BLIPBenchmark()
    blip_vqa_model = BLIPVQABenchmark() # <-- CHANGE THIS LINE
    print("All models loaded.")

    # --- 3. Run Benchmark Loop ---
    all_results = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        print(f"\n--- Processing: {image_name} ---")

        # --- Run CLIP ---
        # (CLIP logic remains the same)
        clip_prompts = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a car", "a photo of a motorcycle"]
        clip_results, clip_latency, clip_memory = clip_model.run_zeroshot_classification(
            image_path=image_path, text_prompts=clip_prompts
        )
        if clip_results:
            top_prediction = max(clip_results, key=clip_results.get)
            all_results.append({
                'model': 'CLIP', 'image': image_name, 'task': 'Zero-Shot Classification',
                'result': f"Top Prediction: {top_prediction} ({clip_results[top_prediction]:.2f})",
                'latency_s': f"{clip_latency:.4f}", 'memory_mb': f"{clip_memory:.2f}"
            })

        # --- Run BLIP (Captioning) ---
        # (BLIP logic remains the same)
        captions, blip_latency, blip_memory = blip_caption_model.run_image_captioning(
            image_path=image_path
        )
        if captions:
            all_results.append({
                'model': 'BLIP-Caption', 'image': image_name, 'task': 'Image Captioning', # Renamed for clarity
                'result': captions, 'latency_s': f"{blip_latency:.4f}", 'memory_mb': f"{blip_memory:.2f}"
            })

        # --- Run BLIP (VQA) ---
        question = "What is in this photo?"
        answer, vqa_latency, vqa_memory = blip_vqa_model.run_vqa(
            image_path=image_path, question=question
        )
        if answer:
            all_results.append({
                'model': 'BLIP-VQA', # <-- CHANGE THIS
                'image': image_name,
                'task': 'Visual Question Answering',
                'result': f'Q: {question} A: {answer}',
                'latency_s': f"{vqa_latency:.4f}",
                'memory_mb': f"{vqa_memory:.2f}"
            })

    # --- 4. Save Results to CSV ---
    # (Saving logic remains the same)
    if not all_results:
        print("No results were generated.")
        return
        
    headers = all_results[0].keys()
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n--- Benchmark Complete ---")
    print(f"Results for {len(image_paths)} images saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()