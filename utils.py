import numpy as np
from scipy.ndimage import gaussian_filter
import xml.etree.ElementTree as ET
import torch

PATCH_SIZE = 256
PIXEL_SIZE_UM = 0.24 
CELL_DIAMETER_UM = 10 
SIGMA = (CELL_DIAMETER_UM / 2) / PIXEL_SIZE_UM 

def create_heatmap(annotations_list, patch_size=PATCH_SIZE, num_classes=2, sigma=SIGMA):
    heatmaps = np.zeros((num_classes, patch_size, patch_size), dtype=np.float32)
    
    for x_rel, y_rel, class_id in annotations_list:
        if 0 <= x_rel < patch_size and 0 <= y_rel < patch_size:
            heatmaps[class_id, y_rel, x_rel] = 1.0 
    
    heatmaps = gaussian_filter(heatmaps, sigma=(0, sigma, sigma))
    scaling_factor = 2 * np.pi * (sigma ** 2)
    heatmaps *= scaling_factor
    
    return heatmaps

def parse_xml_annotations(xml_path, classes={'lymphocyte': 0, 'monocyte': 1}):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

    except ET.ParseError:
        return []
    
    current_slide_annotations = []
    
    for annotation in root.findall('.//Annotation'):
        group = annotation.get('PartOfGroup') 
        
        if group is None:
            continue
            
        clean_group = group.strip().lower()
        if 'lymph' in clean_group:
            class_name = 'lymphocyte'
            class_id = classes[class_name]

        elif 'monoc' in clean_group:
            class_name = 'monocyte'
            class_id = classes[class_name]

        else:
            continue

        for coordinates_tag in annotation.findall('.//Coordinates'):
                for coordinate in coordinates_tag.findall('.//Coordinate'):
                    try:
                        x = int(float(coordinate.get('X')))
                        y = int(float(coordinate.get('Y')))
                        current_slide_annotations.append({
                            'x': x, 
                            'y': y, 
                            'class_id': class_id, 
                            'class_name': class_name
                        })

                    except (ValueError, TypeError):
                        continue
                    
    return current_slide_annotations

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    heatmaps = torch.stack([item[1] for item in batch])
    
    return images, heatmaps