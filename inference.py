import torch
import numpy as np
import openslide
from skimage.feature import peak_local_max
import os
import json
from train import build_model
from utils import PATCH_SIZE 
from torchvision import transforms as T
from tqdm import tqdm

PATCH_SIZE = 256
PIXEL_SIZE_MM = 0.00024
SPACING_VALUE = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '/opt/ml/model/model.pth' 
TEST_SLIDE_DIR = '/input/'
OUTPUT_BASE_DIR = '/output/' 
STRIDE = 128

def inference_wsi(model, slide_path):
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"Could not open slide: {slide_path}. Error: {e}")
        return None

    width, height = slide.level_dimensions[0]
    num_classes = 2
    full_heatmap = np.zeros((num_classes, height, width), dtype=np.float32)
    overlap_counts = np.zeros((height, width), dtype=np.uint8)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    for y in tqdm(range(0, height - PATCH_SIZE + 1, STRIDE), desc="Sliding Window Y"):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch_pil = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch_tensor = transform(patch_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_heatmap = model(patch_tensor).cpu().squeeze(0).numpy()

            full_heatmap[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred_heatmap
            overlap_counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
    
    slide.close()

    overlap_counts_3d = np.stack([overlap_counts] * num_classes, axis=0)
    full_heatmap = np.divide(full_heatmap, overlap_counts_3d, 
                             out=np.zeros_like(full_heatmap), 
                             where=overlap_counts_3d != 0)
    
    return full_heatmap

def post_process(full_heatmap):
    MIN_DISTANCE = 8 
    CONFIDENCE_THRESHOLD = 0.3 

    detections = []
    class_map = {0: "lymphocyte", 1: "monocyte"}

    for class_id in range(2):
        heatmap = full_heatmap[class_id]
        coords = peak_local_max(
            heatmap, 
            min_distance=MIN_DISTANCE, 
            threshold_abs=CONFIDENCE_THRESHOLD,
            exclude_border=False
        )

        for y, x in coords:
            confidence = float(heatmap[y, x])
            detections.append({
                "name": class_map[class_id],
                "x": int(x),
                "y": int(y),
                "probability": confidence
            })
            
    return detections

def save_submission(detections, slide_id, output_base_dir):
    lympho_detections = [d for d in detections if d['name'] == 'lymphocyte']
    mono_detections = [d for d in detections if d['name'] == 'monocyte']
    all_detections = lympho_detections + mono_detections

    os.makedirs(output_base_dir, exist_ok=True) 

    output_files = {
        "lymphocytes": (lympho_detections, os.path.join(output_base_dir, "detected-lymphocytes.json")),
        "monocytes": (mono_detections, os.path.join(output_base_dir, "detected-monocytes.json")),
        "inflammatory-cells": (all_detections, os.path.join(output_base_dir, "detected-inflammatory-cells.json"))
    }

    def format_output_json(name, detection_list):
        formatted_points = []
        for i, d in enumerate(detection_list):
            x_mm = float(d['x']) * PIXEL_SIZE_MM
            y_mm = float(d['y']) * PIXEL_SIZE_MM
            
            formatted_points.append({
                "name": f"Point {i+1}",
                "point": [x_mm, y_mm, SPACING_VALUE], 
                "probability": float(d['probability'])
            })
            
        return {
            "name": name,
            "type": "Multiple points",
            "points": formatted_points,
            "version": { "major": 1, "minor": 0 }
        }
    
    total_saved = 0
    for name, (data, output_path) in output_files.items():
        output_json = format_output_json(name, data)
        
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=4)
            
        total_saved += len(data)
        
    print(f"Saved submission files for {slide_id}. Total detections: {len(all_detections)}")

if __name__ == '__main__':
    print("Initializing model...")
    model = build_model().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model weights not found at {MODEL_PATH}. Check your Docker setup.")
        exit(1)
        
    model.eval()
    print("Model loaded successfully.")

    try:
        slide_filenames = [f for f in os.listdir(TEST_SLIDE_DIR) if f.endswith('.tif')]
        if not slide_filenames:
            print(f"No TIFF slides found in the input directory: {TEST_SLIDE_DIR}")
            exit(1)

        slide_filename = slide_filenames[0]
        slide_path = os.path.join(TEST_SLIDE_DIR, slide_filename)
        slide_id = slide_filename.split('_PAS_CPG.tif')[0]
        
    except FileNotFoundError:
        print(f"Input directory not found: {TEST_SLIDE_DIR}. Ensure Docker paths are correct.")
        exit(1)

    print(f"\n--- Starting Processing for Slide: {slide_id} ---")

    full_heatmap = inference_wsi(model, slide_path) 
    if full_heatmap is None:
        exit(1)

    detections = post_process(full_heatmap) 

    save_submission(detections, slide_id, output_base_dir=OUTPUT_BASE_DIR)

    print("\n✅ Inference complete and JSON files saved to /output/.")