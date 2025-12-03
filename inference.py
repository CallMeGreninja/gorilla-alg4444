import torch
import torch.nn as nn
import numpy as np
import openslide
from skimage.feature import peak_local_max
import os
import json
import glob
from torchvision import transforms as T
from tqdm import tqdm
import segmentation_models_pytorch as smp
import time

PATCH_SIZE = 256
PIXEL_SIZE_MM = 0.00024
SPACING_VALUE = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STRIDE = 256
INFERENCE_BATCH_SIZE = 64
MIN_TISSUE_THRESHOLD = 0.3

POSSIBLE_MODEL_PATHS = [
    '/opt/ml/model/best_model.pth',
    '/app/best_model.pth',
    './best_model.pth',
    '/opt/algorithm/best_model.pth'
]

INPUT_IMAGE_DIR = '/input/images/kidney-transplant-biopsy-wsi-pas/'
INPUT_MASK_DIR = '/input/images/tissue-mask/'
OUTPUT_DIR = '/output/'

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation='sigmoid',
    )
    return model

def find_model_path():
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path

    for root, dirs, files in os.walk('/'):
        for file in files:
            if file == 'best_model.pth':
                found_path = os.path.join(root, file)
                print(f"Found model at: {found_path}")
                return found_path
    
    raise FileNotFoundError("Model weights not found in any expected location")

def get_input_files():
    print(f"Looking for WSI in: {INPUT_IMAGE_DIR}")
    print(f"Looking for mask in: {INPUT_MASK_DIR}")

    wsi_files = glob.glob(os.path.join(INPUT_IMAGE_DIR, '*.tif*'))
    if not wsi_files:
        wsi_files = glob.glob('/input/**/*PAS*.tif*', recursive=True)

    mask_files = glob.glob(os.path.join(INPUT_MASK_DIR, '*.tif*'))
    if not mask_files:
        mask_files = glob.glob('/input/**/*mask*.tif*', recursive=True)
    
    wsi_path = wsi_files[0] if wsi_files else None
    mask_path = mask_files[0] if mask_files else None
    
    print(f"WSI path: {wsi_path}")
    print(f"Mask path: {mask_path}")
    
    return wsi_path, mask_path

def get_roi_bounds(mask_path):
    if not mask_path or not os.path.exists(mask_path):
        return None
    
    try:
        mask_slide = openslide.OpenSlide(mask_path)
        level = min(3, mask_slide.level_count - 1)
        mask_dims = mask_slide.level_dimensions[level]
        
        mask_region = mask_slide.read_region((0, 0), level, mask_dims).convert('L')
        mask_np = np.array(mask_region)
        
        rows, cols = np.where(mask_np > 0)
        
        if len(rows) > 0:
            scale = mask_slide.level_downsamples[level]
            bounds = {
                'xmin': int(cols.min() * scale),
                'xmax': int(cols.max() * scale),
                'ymin': int(rows.min() * scale),
                'ymax': int(rows.max() * scale),
                'mask_np': mask_np,
                'scale': scale,
                'offset_x': cols.min(),
                'offset_y': rows.min()
            }
            mask_slide.close()
            return bounds
        
        mask_slide.close()
    except Exception as e:
        print(f"Error processing mask: {e}")
    
    return None

def is_tissue_patch(patch_np, threshold=MIN_TISSUE_THRESHOLD):
    gray = np.mean(patch_np, axis=2)
    tissue_pixels = np.sum(gray < 220)
    tissue_ratio = tissue_pixels / (patch_np.shape[0] * patch_np.shape[1])
    return tissue_ratio > threshold

def check_mask_overlap(x, y, bounds, patch_size=PATCH_SIZE):
    if bounds is None:
        return True

    in_x_bounds = bounds['xmin'] <= x <= bounds['xmax'] - patch_size
    in_y_bounds = bounds['ymin'] <= y <= bounds['ymax'] - patch_size
    
    return in_x_bounds and in_y_bounds

def process_slide(model, slide_path, mask_path=None):
    print(f"Opening slide: {slide_path}")
    
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        print(f"ERROR: Cannot open slide: {e}")
        return []
    
    width, height = slide.level_dimensions[0]
    print(f"Slide dimensions: {width} x {height}")

    bounds = get_roi_bounds(mask_path)
    
    if bounds:
        start_x, end_x = bounds['xmin'], bounds['xmax']
        start_y, end_y = bounds['ymin'], bounds['ymax']
        print(f"Using ROI bounds: x[{start_x}:{end_x}], y[{start_y}:{end_y}]")
    else:
        start_x, start_y = 0, 0
        end_x, end_y = width, height
        print("Processing entire slide (no mask)")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    all_detections = []

    print("Generating patch coordinates...")
    patch_coords = []
    skipped_by_mask = 0
    skipped_by_tissue = 0
    
    for y in range(start_y, end_y - PATCH_SIZE + 1, STRIDE):
        for x in range(start_x, end_x - PATCH_SIZE + 1, STRIDE):
            patch_coords.append((x, y))
    
    print(f"Generated {len(patch_coords)} patches from ROI")

    start_time = time.time()
    processed_patches = 0
    
    for batch_idx in tqdm(range(0, len(patch_coords), INFERENCE_BATCH_SIZE), desc="Inference"):
        batch_patches = []
        batch_positions = []

        for x, y in patch_coords[batch_idx:batch_idx + INFERENCE_BATCH_SIZE]:
            try:
                patch_pil = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
                patch_np = np.array(patch_pil)

                if not is_tissue_patch(patch_np):
                    skipped_by_tissue += 1
                    continue
                
                patch_tensor = transform(patch_pil)
                batch_patches.append(patch_tensor)
                batch_positions.append((x, y))
            except Exception as e:
                continue
        
        if not batch_patches:
            continue

        batch_tensor = torch.stack(batch_patches).to(DEVICE)
        
        with torch.no_grad():
            pred_heatmaps = model(batch_tensor).cpu().numpy()

            if batch_idx == 0:
                print(f"\nDEBUG: Heatmap shape: {pred_heatmaps.shape}")
                print(f"DEBUG: Heatmap range: [{pred_heatmaps.min():.4f}, {pred_heatmaps.max():.4f}]")
                print(f"DEBUG: Heatmap mean: {pred_heatmaps.mean():.4f}")

        for i, (x, y) in enumerate(batch_positions):
            pred_heatmap = pred_heatmaps[i]
            
            for class_id, class_name in enumerate(["lymphocyte", "monocyte"]):
                heatmap = pred_heatmap[class_id]

                if heatmap.max() < 0.3:
                    continue

                local_peaks = peak_local_max(
                    heatmap,
                    min_distance=10,
                    threshold_abs=0.3,
                    exclude_border=False
                )

                for p_y, p_x in local_peaks:
                    global_x = x + p_x
                    global_y = y + p_y
                    confidence = float(heatmap[p_y, p_x])
                    
                    all_detections.append({
                        "name": class_name,
                        "x": int(global_x),
                        "y": int(global_y),
                        "probability": confidence
                    })
        
        processed_patches += len(batch_patches)
    
    slide.close()
    
    elapsed = time.time() - start_time
    print(f"✅ Processed {processed_patches} patches in {elapsed:.1f}s ({processed_patches/elapsed:.1f} patches/s)")
    print(f"Skipped: {skipped_by_mask} by mask, {skipped_by_tissue} by tissue check")
    print(f"Found {len(all_detections)} detections")
    
    return all_detections

def save_detections(detections, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    lymphocytes = [d for d in detections if d['name'] == 'lymphocyte']
    monocytes = [d for d in detections if d['name'] == 'monocyte']
    
    print(f"Total detections: {len(detections)} (Lymphocytes: {len(lymphocytes)}, Monocytes: {len(monocytes)})")

    output_configs = [
        ("detected-lymphocytes.json", "lymphocytes", lymphocytes),
        ("detected-monocytes.json", "monocytes", monocytes),
        ("detected-inflammatory-cells.json", "inflammatory-cells", detections)
    ]
    
    for filename, name, data in output_configs:
        formatted_points = []
        for i, d in enumerate(data):
            x_mm = float(d['x']) * PIXEL_SIZE_MM
            y_mm = float(d['y']) * PIXEL_SIZE_MM
            
            formatted_points.append({
                "name": f"Point {i+1}",
                "point": [x_mm, y_mm, SPACING_VALUE],
                "probability": d['probability']
            })

        output_json = {
            "name": name,
            "type": "Multiple points",
            "points": formatted_points,
            "version": {"major": 1, "minor": 0}
        }

        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=4)
        
        print(f"Saved {len(formatted_points)} points to {output_path}")

def main():
    print("=" * 60)
    print("🚀 MONKEY Challenge Inference (OPTIMIZED)")
    print("=" * 60)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Batch size: {INFERENCE_BATCH_SIZE}, Stride: {STRIDE}, Patch size: {PATCH_SIZE}")

    try:
        model_path = find_model_path()
        print(f"Loading model from: {model_path}")
        
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model: {e}")
        save_detections([], OUTPUT_DIR)
        return
    
    try:
        wsi_path, mask_path = get_input_files()
        
        if not wsi_path or not os.path.exists(wsi_path):
            print(f"ERROR: No WSI found. Saving empty results.")
            save_detections([], OUTPUT_DIR)
            return
        
    except Exception as e:
        print(f"ERROR finding input files: {e}")
        save_detections([], OUTPUT_DIR)
        return
    
    try:
        detections = process_slide(model, wsi_path, mask_path)
    except Exception as e:
        print(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        detections = []

    try:
        save_detections(detections, OUTPUT_DIR)
        print("✅ Inference completed successfully")
    except Exception as e:
        print(f"ERROR saving results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()