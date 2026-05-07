### Project-Specific Guidelines - AAI Chest X-ray Pneumonia Project

#### 1. Coding Style & Format
- **Notebook Format**: All code must be written in Jupyter Notebooks (`.ipynb`) using the `#%%` cell delimiter for IDE compatibility.
- **Documentation**: 
  - Every code cell must have comments inside explaining the logic.
  - Markdown cells must be placed between code cells to explain the theory, logic, and provide easy understanding.
- **Library Usage**:
  - Use `OpenCV` for initial artifact detection and removal.
  - Use `PyTorch` for model training and data handling.
  - Use a pretrained `U-Net` for lung segmentation in Run 2.

#### 2. Research Methodology
- **Framework**: Nunamaker et al. multi-methodological research method.
- **Research Question (Q1)**: "Does removing non-anatomical artifacts from chest X-rays improve pneumonia classification performance?"
- **Experimental Control**: When changing hyperparameters, ensure they are changed identically across datasets (unprocessed vs. cleaned) to maintain comparability.

#### 3. Data Handling
- **Split Strategy**: Maintain the 80/10/10 train/val/test split.
- **Class Imbalance**: Always address the imbalance between `NORMAL` and `PNEUMONIA` classes using `WeightedRandomSampler` or loss weights.
- **Preprocessing Pipeline**:
  - Padding to maintain aspect ratio before resizing.
  - Resizing to 224x224 pixels.
  - Batch normalization (to be evaluated).
  - Light augmentation (rotation, zoom, shift).

#### 4. Model Training & Evaluation
- **Tracking**: Monitor training loss, validation loss, accuracy, recall, and precision per epoch.
- **Hyperparameter Tuning**: Use random or grid search.
- **Evaluation Metrics**:
  - Confusion Matrix.
  - AUC and ROC curves.
  - Threshold tuning.
  - 5-fold Cross-validation.

#### 5. Artifact Removal (Run 2)
- Focus on detecting and removing non-anatomical markers (e.g., Letter 'R').
- Primary method: Lung segmentation via U-Net to isolate lung fields.
- Secondary methods: Fixed cropping, intensity-based cropping, corner masking, or inpainting.
