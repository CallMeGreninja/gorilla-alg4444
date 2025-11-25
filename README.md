# 🐵 **MONKEY Challenge: Kidney Inflammation Detection**


## 📋 **Overview**

This repository contains a deep learning solution for the MONKEY Challenge (Machine-learning for Optimal detection of iNflammatory cells in the KidnEY): https://monkey.grand-challenge.org/

The goal is to automatically detect and classify inflammatory cells (Lymphocytes and Monocytes) in Whole Slide Images (WSI) of kidney transplant biopsies. This solution utilizes a U-Net architecture with a ResNet34 encoder to perform heatmap regression, effectively locating cell centers even in dense tissue regions.


## ✨ **Key Features**

Architecture: U-Net with a pre-trained ResNet34 backbone.

Methodology: Gaussian Heatmap Regression (predicting cell centers vs. binary segmentation).

Data Handling: Efficient WSI processing using OpenSlide with systematic tiling (sliding window) to handle gigapixel images.

Optimization: Custom Weighted MSE Loss to handle extreme class imbalance (background vs. small cells).

Augmentation: Robust pipeline using Albumentations (color jitter, flips, rotations) to handle stain variability.

Inference: Fast inference pipeline with peak detection logic compatible with Grand Challenge Docker requirements.


## 🛠️ **Installation**

1. Prerequisites

Python 3.8+

NVIDIA GPU (Recommended: 8GB+ VRAM) with CUDA installed.

OpenSlide Binaries:

Windows: Download binaries from OpenSlide and add to PATH.

macOS: brew install openslide

Linux: sudo apt-get install libopenslide0

2. Clone Repository

git clone [https://github.com/YourUsername/monkey-challenge.git](https://github.com/YourUsername/monkey-challenge.git)
cd monkey-challenge


3. Install Dependencies

It is highly recommended to use a virtual environment.

For CUDA Users (Windows/Linux):
First, install the PyTorch version matching your system's CUDA version (e.g., CUDA 12.1):

pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)


Then install the rest of the requirements:

pip install -r requirements.txt



## 📂 **Data**

The project requires a specific directory structure. You must download the data from the AWS Open Data Registry: https://registry.opendata.aws/monkey/

monkey/  
├── data/  
│   ├── train/  
│   │   ├── annotations/  
│   │   │   └── xml/             # Raw XML annotations from ASAP  
│   │   ├── images/  
│   │   │   ├── pas-cpg/         # .tif WSI files  
│   │   │   └── tissue-masks/    # .tif Tissue masks  
│   │   └── ...  
│   └── validation/  
│       ├── annotations/  
│       │   └── xml/             # Validation XMLs  
│       ├── images/  
│       │   ├── pas-cpg/         # Validation .tif WSI files  
│       │   └── tissue-masks/    # Validation .tif Tissue masks  
│       └── ... (Ground Truth JSONs in root of validation)  
├── model_weights/               # Saved checkpoints  
├── dataset.py  
├── train.py  
├── inference.py  
└── utils.py  
  


## 🚀 **Usage**

Training

To train the model from scratch. This script uses systematic tiling to generate ~240,000 training patches per epoch.

python train.py


Configuration: Adjust BATCH_SIZE, LEARNING_RATE, and NUM_EPOCHS at the top of train.py.

Checkpoints: The best model (lowest validation loss) is automatically saved to ./model_weights/best_model.pth.

Inference (Local)

To run inference on the validation set and generate JSON predictions:

python inference.py



## 🧠 **Technical Details**

Data Sampling

Instead of randomly sampling patches, I employ a Systematic Tiling strategy. I iterate through the valid tissue area (defined by tissue masks) using a stride equal to the patch size ($256\times256$). This ensures the model sees 100% of the tissue, including empty background areas, reducing false positives.

Loss Function

I utilize a Weighted MSE Loss. Since cells occupy <1% of the pixels, a standard MSE loss leads to model collapse (predicting all zeros). I apply a weight factor (e.g., $100\times$) to pixels containing cell gaussian peaks to force the model to learn features.
