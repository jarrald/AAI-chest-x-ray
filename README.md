# AAI Chest X-ray Pneumonia Classification

This project is part of an Advanced AI semester project focused on pneumonia detection from chest X-ray images, following the **Nunamaker et al. multi-methodological research method**.

## Research Question
**Q1: Does removing non-anatomical artifacts (e.g., Letter 'R') from chest X-rays improve pneumonia classification performance?**

## Project Overview
The project is divided into multiple runs to compare the performance of a CNN model trained on unprocessed data versus data where non-anatomical artifacts have been removed.

### Current Progress: Run 1
We have completed the initial setup, data visualization, and preprocessing for Run 1 (no artifact cleaning).
- **Dataset Split**: 80/10/10 (Train/Val/Test).
- **Class Imbalance**: Handled using `WeightedRandomSampler` to address the higher number of healthy lung images compared to pneumonia images.
- **Preprocessing**: 
  - Padding and resizing to 224x224.
  - Normalization and light augmentation.
- **Model Training**: *Pending*.

### Planned: Run 2 (Artifact Cleaning)
In Run 2, we will detect and remove non-anatomical artifacts using:
- **OpenCV** for marker detection.
- **U-Net** for lung segmentation (isolating lung fields to remove all artifacts).

## Technical Stack
- **Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Segmentation**: U-Net
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## How to use this project
The code is organized in Jupyter Notebooks with the `#%%` format. 
- `no_letters.ipynb`: Contains the workflow for Run 1.
- Each notebook includes theoretical markdown explanations between code cells.

## Methodology Notes
To ensure comparability, all hyperparameters will be kept identical between Run 1 and Run 2 when training the models.
