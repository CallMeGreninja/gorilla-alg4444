import torch
from torch.utils.data import Dataset
import numpy as np
import openslide
from PIL import Image
from utils import create_heatmap, PATCH_SIZE, parse_xml_annotations 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import os
import json
import re 
from tqdm import tqdm 

class MonkeyDataset(Dataset):
    def __init__(self, data_root_dir, split='train'):
        self.data_root_dir = data_root_dir
        self.split = split
        self.patch_size = PATCH_SIZE
        
        self.pas_cpg_dir = os.path.join(data_root_dir, 'images', 'pas-cpg')
        self.xml_dir = os.path.join(data_root_dir, 'annotations', 'xml')
        self.mask_dir = os.path.join(data_root_dir, 'images', 'tissue-masks')
        
        self.annotations = self._load_all_annotations(self.xml_dir) 
        self.roi_bounds = self._load_roi_bounds(self.mask_dir)

        self.samples = self._create_patch_samples(self.roi_bounds)
        self.transform = self._get_augmentations(split)
        
    def _load_all_annotations(self, xml_dir):
        all_annotations = {}
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
        
        print(f"Loading annotations from {len(xml_files)} XML files...")

        for filename in tqdm(xml_files):
            xml_path = os.path.join(xml_dir, filename)
            slide_id = os.path.splitext(filename)[0]
            
            annotations_list = parse_xml_annotations(xml_path)
            
            if annotations_list:
                all_annotations[slide_id] = annotations_list
                
        print(f"Successfully loaded annotations for {len(all_annotations)} slides.")

        return all_annotations
    
    def _load_roi_bounds(self, tissue_mask_dir):
        roi_bounds = {}
        mask_files = [f for f in os.listdir(tissue_mask_dir) if f.endswith('.tif')]
        
        print(f"Calculating ROI bounds from {len(mask_files)} TIF masks...")

        for mask_filename in tqdm(mask_files):
            mask_path = os.path.join(tissue_mask_dir, mask_filename)
            
            match = re.match(r'(.+)_mask\.tif', mask_filename)

            if not match: continue
            slide_id = match.group(1) 
            
            mask_slide = None

            try:
                mask_slide = openslide.OpenSlide(mask_path)

                level_to_read = min(3, mask_slide.level_count - 1) 
                mask_dimensions = mask_slide.level_dimensions[level_to_read]
                
                mask_patch = mask_slide.read_region((0, 0), level_to_read, mask_dimensions)
                mask_data = np.array(mask_patch.convert("L")) 
                
                y_coords, x_coords = np.where(mask_data > 0) 
                
                if len(x_coords) > 0:
                    xmin_low, ymin_low = x_coords.min(), y_coords.min()
                    xmax_low, ymax_low = x_coords.max(), y_coords.max()
                    
                    downsample_factor = mask_slide.level_downsamples[level_to_read]
                    
                    xmin = int(xmin_low * downsample_factor)
                    ymin = int(ymin_low * downsample_factor)
                    xmax = int(xmax_low * downsample_factor)
                    ymax = int(ymax_low * downsample_factor)
                    
                    roi_bounds[slide_id] = {'xmin': xmin, 'ymin': ymin,
                                            'xmax': xmax, 'ymax': ymax}
                
            except Exception as e:
                print(f"Warning: Error processing mask {mask_filename}. Skipping. Error: {e}")

            finally:
                if mask_slide is not None:
                    mask_slide.close()
                
        print(f"Successfully computed ROI bounds for {len(roi_bounds)} slides from TIF masks.")

        return roi_bounds
    
    def _create_patch_samples(self, roi_bounds):
        samples = []
        STRIDE = self.patch_size

        for slide_id, bounds in roi_bounds.items():
            xmin, ymin = bounds['xmin'], bounds['ymin']
            xmax, ymax = bounds['xmax'], bounds['ymax']

            for x_global in range(xmin, xmax - self.patch_size + 1, STRIDE):
                for y_global in range(ymin, ymax - self.patch_size + 1, STRIDE):
                    samples.append((slide_id, x_global, y_global))

        return samples

    def _get_augmentations(self, split):
        base_augs = [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
            ToTensorV2() 
        ]
        
        if split == 'train':
            return A.Compose(base_augs)
        
        else:
            return A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        slide_id, x_global, y_global = self.samples[index]
        
        slide_path = os.path.join(self.pas_cpg_dir, f"{slide_id}_PAS_CPG.tif")
        slide = None

        try:
            slide = openslide.OpenSlide(slide_path)
            pas_patch_pil = slide.read_region((x_global, y_global), 0, (self.patch_size, self.patch_size))
            pas_patch = np.array(pas_patch_pil.convert("RGB"))

        except Exception as e:
            print(f"Error reading slide {slide_id}: {e}. Retrying with random index.")
            return self.__getitem__(random.randint(0, len(self.samples) - 1)) 
        
        finally:
            if slide is not None:
                slide.close()

        patch_annotations = []
        for dot in self.annotations.get(slide_id, []):
            x_dot, y_dot, class_id = dot['x'], dot['y'], dot['class_id']
            
            if (x_global <= x_dot < x_global + self.patch_size) and \
               (y_global <= y_dot < y_global + self.patch_size):
                
                x_local = x_dot - x_global
                y_local = y_dot - y_global
                patch_annotations.append([x_local, y_local, class_id])
                
        target_heatmap = create_heatmap(patch_annotations, num_classes=2)

        if self.transform:
            augmented = self.transform(image=pas_patch, masks=[target_heatmap[i] for i in range(2)])
            pas_patch = augmented['image'] 
            target_heatmap = torch.stack(augmented['masks'])

        return pas_patch, target_heatmap