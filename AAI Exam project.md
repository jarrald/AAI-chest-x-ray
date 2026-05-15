**Mandatory 2**   
**research question**   
Does removing non-anatomical artifacts from chest X-rays improve pneumonia classification performance?

**Proces**   
Following Nunamaker et al.’s multi-methodological research process, the project was structured as an iterative process between observation, systems development, and experimentation. First, the original chest X-ray dataset was explored to understand its structure, class distribution, and visual characteristics. Because the original validation set was very small, the dataset was reorganized into an 80/10/10 split for training, validation, and testing.

The dataset was visualized through class distribution plots and sample images from both NORMAL and PNEUMONIA classes. These observations informed the development of the preprocessing pipeline. Images were converted to grayscale with three channels, padded to avoid distortion, resized to 224x224 pixels, and augmented during training using small rotations, shifts, and zooming. A weighted random sampler was used to address class imbalance.

The first experiment trained a baseline model on the original images, including non-anatomical artifacts such as L/R markers. This baseline will later be compared with a model trained on a cleaned version of the dataset, allowing us to evaluate whether artifact removal improves classification performance.

**synopsis**  
Train CNN on both unprocessed dataset and a dataset where non-anatomical artifacts (Letter R for patient’s right side) is removed. When you change the hyper-parameters, change them exactly the same way when you train the  model with another dataset, so that results remain entirely comparable.

How can we detect the non-anatomical artifacts? What is the best/most time effective method of removing them?

How to deal with class imbalance? There are significantly more images of healthy lungs than there are of those with lungs with pneumonia. 

**Choice of model**

We have explored different deep learning models, such as ResNet-18 and DenseNet-121. These models are pre-trained on large image datasets, making them strong baseline choices for image classification tasks.

**Choice of how to test performance** 

To evaluate the model’s performance, we use metrics such as accuracy, precision, recall, and F1-score. These metrics help us understand how well the model performs, especially when working with imbalanced datasets. 

**Progress so far:**

We’re finished with the first run, we have the following steps

1. Load test data (Data is already split but we want a different split)  
2. Data split into 80/10/10

**Mandatory 2:**

Requirements:

- Problem formulation  
- Synopsis  
- Considerations regarding models  
- Considerations regarding performance metrics

**Class imbalance source**

**Nunamaker et al. multi-methodological research method**

Q1: Does removing non-anatomical artifacts from chest X-rays improve pneumonia classification performance?

Train CNN on both unprocessed dataset and a dataset where non-anatomical artifacts (Letter R for patient’s right side) is removed. When you change the hyper-parameters, change them exactly the same way when you train the  model with another dataset, so that results remain entirely comparable.

How can we detect the non-anatomical artifacts? What is the best/most time effective method of removing them?

How to deal with class imbalance? There are significantly more images of healthy lungs than there are of those with lungs with pneumonia. 

Use : OpenCV to find R or other artifacts to later remove them.

Use a pretrained lung segmentation model (e.g., U-Net) to isolate lung fields and train only on that mask. Removes all non-anatomical artifacts reliably. 

## Links

[https://github.com/seungjunlee96/U-Net\_Lung-Segmentation](https://github.com/seungjunlee96/U-Net_Lung-Segmentation)  
[https://kea-aai-2026-1-da1461.gitlab.io/mandatory-assignments.html](https://kea-aai-2026-1-da1461.gitlab.io/mandatory-assignments.html)  
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
[https://github.com/jarrald/AAI-chest-x-ray](https://github.com/jarrald/AAI-chest-x-ray) 

## Notes

Dataset has a strange train/val/test split

Steps/fremgang med projektet:  
**Run 1, no cleaning of letters and artifacts:**

1. Data visualization \- done   
   We visualize the chest X-ray dataset to understand its structure and composition:  
   1\. Dataset Split (80/10/10)\*\* \- Bar chart showing how many images are in train/val/test for each class (NORMAL vs PNEUMONIA)  
   2\. Class Distribution\*\* \- Pie/bar chart showing the overall class imbalance \- there are significantly more NORMAL images than PNEUMONIA  
   3\. Sample Images\*\* \- Grid of 10 sample X-rays (5 NORMAL, 5 PNEUMONIA) from the training set  
   4\. Preprocessed Images\*\* \- Examples of how images look after transformations (padding, resize, normalization, augmentation)  
   This helps us verify the data is loaded correctly and understand what the model will see during training.  
     
2. Preprocessing \- done   
   1. Add padding to rectangles to avoid stretching when images are resized  
   2. Resize images to 224\*224 pixels, which is the standard for these types of image classification models (large enough to have relevant features, and not too big so training is too slow)  
   3. Light augmentation by adding slight rotation, zoom and shift up/down to images  
3. Handle class imbalance \- done  
   1. Instead of sampling images randomly in order, it assigns each image a weight based on its class, Images from underrepresented classes (fewer images) get higher weights sampled more often  
   2. Images from overrepresented classes (many images) get lower weights → sampled less often  
   3. Result: each batch has roughly equal amounts of NORMAL and PNEUMONIA  
   4. We do it because we don't have a lot of normal image and therefore use weight on normal.   
4. Scaling? Different size images \- done  
5. Batch normalization \- done  
6. Data augmentation \- done  
7. Train the data \- done  
8. Cross validation \- (fjernet)  
9. Confusion Matrix \- done  
10. Track per epoch:  
    1. Training loss (done)  
    2. Validation loss (done)  
11. Metrics (accuracy, recall, precision)  
    1. First run was bad, precision went down, most values stayed the same and loss didn’t go down  
12. See if the data is underfitted/overfitted  
    1. Bad training and validation scores \= underfitted  
    2. Good training but bad validation scores \= overfitted  
13. Hyperparameter tuning (Important) random/grid search  
    1. We use random search for these values  
       1. Learning rate  
       2. Dropout rate  
       3. Weight decay  
       4. Batch size  
14. Make a loss plot (done)  
15. Use different optimizers (Skipped for now)  
16. We use AdamW because of batch normalization (default good start)  
17. SGD (better generalization sometimes)  
18. Evaluate  
19. AUC og ROC curve (Done)  
20. Threshold tuning  
21. Cross-validation (K-fold, 5-fold)  
22. Save model  
23. Kilder 

 Nunamaker-processen:

- Nunamaker Jr, J. F., Chen, M., & Purdin, T. D. M. (1990). Systems development in information systems research. Journal of Management Information Systems, 7(3), 89–106.

Class imbalance:

- He, H., & Garcia, E. A. (2009). *Learning from imbalanced data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.    
- Krawczyk, B. (2016). *Learning from imbalanced data: open challenges and future directions*. Progress in Artificial Intelligence, 5, 221–232. 

Data augmentation 

- Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data, 6, 60\. 

Resizing til 224x224 

- PyTorch Torchvision documentation. *ResNet18 models and pretrained weights.* 

CNN / image classification generelt 

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks*. Advances in Neural Information Processing Systems. 

**Cleaning of dataset on run 2**

Common methods, ordered from simplest to strongest:

1. Fixed cropping  
    Crop a constant margin (or center-crop) to remove borders where markers/text usually sit. Fast, but can cut useful anatomy if not tuned.  
2. Intensity/threshold-based cropping  
    Detect the body region and crop to its bounding box (as in contour/threshold approach). Removes black borders and most corner text.  
3. Corner masking  
    Zero out (black) fixed corner regions where “R/L” markers typically appear. Very cheap, avoids affecting central anatomy.  
4. Text/marker detection \+ inpainting  
    Detect high-contrast small components (letters, labels) via connected components or edge detection, then remove them with inpainting (e.g., OpenCV inpaint). More precise, moderate effort.  
5. Lung-region cropping via heuristics  
    Use simple rules (e.g., keep central vertical band, ignore outer margins) to approximate lung area. Crude but often effective.  
6. Lung segmentation (best)  
    Use a pretrained lung segmentation model (e.g., U-Net) to isolate lung fields and train only on that mask. Removes all non-anatomical artifacts reliably.  
7. Data augmentation to neutralize artifacts  
    Random crops, shifts, and flips so the model cannot rely on fixed marker positions. This doesn’t remove artifacts but reduces their impact.  
8. Frequency/domain filtering (rare)  
    Suppress sharp text edges using filters. Not common; can degrade image quality.

# Introduction

# Problem formulation

# Methods

# Analysis

# Findings

# Conclusion

# References

Alright—here’s the full, end-to-end pipeline for building a binary CNN classifier for lung images (e.g., sick vs healthy). This is not just “train a model”—it’s the entire lifecycle from raw data to deployment, including the subtle stuff people often skip.

---

# **1\. Problem Definition & Constraints**

Before touching data:

* Define task: binary classification (sick vs healthy)  
* Clarify:  
  * Image modality (X-ray, CT, MRI)  
  * Input size constraints (GPU memory, resolution)  
  * Class balance (often skewed in medical datasets)  
  * Performance metric priorities:  
    * Accuracy is not enough  
    * Use **recall/sensitivity** (missed disease is costly)  
    * Also track **precision, F1, AUC-ROC**

---

# **2\. Data Acquisition & Organization**

### **Structure your dataset**

dataset/  
 train/  
   sick/  
   healthy/  
 val/  
   sick/  
   healthy/  
 test/  
   sick/  
   healthy/

### **Split strategy**

Train: 80% / Validation: 10% / Test: 10%

⚠️ Critical:

* Split **by patient**, not image (avoid leakage)  
* Keep class distribution consistent (stratified split)

---

# **3\. Data Inspection & Exploration (EDA)**

### **3.1 Visual inspection**

* Plot random samples per class  
* Look for:  
  * Artifacts (text overlays, markers)  
  * Different orientations  
  * Resolution inconsistencies

### **3.2 Statistical inspection**

* Image sizes distribution  
* Pixel intensity histogram  
* Class imbalance

### **3.3 Check for:**

* Duplicates  
* Corrupted files  
* Label noise

---

# **4\. Data Cleaning**

* Remove corrupted images  
* Remove duplicates (hashing or perceptual similarity)  
* Fix incorrect labels (manual or semi-automatic review)  
* Normalize formats (PNG/JPEG consistency)

---

# **5\. Preprocessing**

### **5.1 Resize**

* Common sizes: 224×224, 256×256  
* Tradeoff:  
  * Smaller \= faster  
  * Larger \= more detail

### **5.2 Normalization**

* Scale pixel values:

   x \= x / 255.0

* Or use dataset mean/std normalization

### **5.3 Grayscale handling**

* If grayscale:  
  * Convert to 3 channels (duplicate) if using pretrained CNNs

---

# **6\. Data Augmentation**

Very important for medical datasets.

Apply **only to training set**:

* Rotation (±10–20°)  
* Horizontal flip (if anatomically valid)  
* Zoom  
* Shift  
* Contrast adjustment  
* Gaussian noise

Avoid:

* Unrealistic distortions

---

# **7\. Dataset Pipeline**

Use efficient loaders:

* PyTorch: Dataset \+ DataLoader  
* TensorFlow: tf.data

Include:

* Batching  
* Shuffling (train only)  
* Prefetching

---

# **8\. Model Design**

## **Option A: Custom CNN**

Typical structure:

Conv → ReLU → Pool  
Conv → ReLU → Pool  
Conv → ReLU → Pool  
Flatten  
Dense → ReLU  
Dropout  
Dense → Sigmoid

## **Option B (Recommended): Transfer Learning**

Use pretrained models:

* ResNet  
* EfficientNet  
* VGG

Steps:

1. Load pretrained backbone  
2. Freeze base layers  
3. Replace classifier head

---

# **9\. Model Compilation**

### **Loss function**

* Binary classification:

   Binary Crossentropy

### **Optimizer**

* Adam (default good start)  
* SGD (better generalization sometimes)

### **Learning rate**

* Typical: 1e-3 → 1e-5

---

# **10\. Training Loop**

Track per epoch:

* Training loss  
* Validation loss  
* Metrics (accuracy, recall, precision)

---

# **11\. Monitoring Training**

### **Plot:**

* Loss vs epoch  
* Accuracy vs epoch

Interpretation:

* Training ↓, Validation ↑ → overfitting  
* Both high → underfitting

---

# **12\. Overfitting / Underfitting Handling**

## **Overfitting fixes**

* Data augmentation ↑  
* Dropout ↑  
* L2 regularization  
* Reduce model size  
* Early stopping  
* More data

## **Underfitting fixes**

* Larger model  
* Train longer  
* Reduce regularization  
* Better features (transfer learning)

---

# **13\. Hyperparameter Tuning**

Tune:

* Learning rate  
* Batch size  
* Optimizer  
* Architecture depth  
* Dropout rate

Methods:

* Grid search  
* Random search  
* Bayesian optimization

---

# **14\. Model Evaluation**

### **On test set ONLY (once)**

Compute:

* Accuracy  
* Precision  
* Recall (very important)  
* F1-score  
* AUC-ROC

---

# **15\. Confusion Matrix**

Shows:

* True positives (TP)  
* False negatives (FN) ← critical in medical  
* False positives (FP)  
* True negatives (TN)

---

# **16\. Threshold Tuning**

Default:

threshold \= 0.5

But:

* Lower threshold → higher recall  
* Adjust depending on medical risk

---

# **17\. ROC Curve & AUC**

* Plot TPR vs FPR  
* AUC close to 1 \= strong model

---

# **18\. Explainability (VERY important in medical AI)**

Use:

* Grad-CAM  
* Saliency maps

Check:

* Is model focusing on lungs?  
* Or cheating via artifacts?

---

# **19\. Cross-Validation (Optional but strong)**

* K-fold (e.g., 5-fold)  
* More reliable performance estimate

---

# **20\. Error Analysis**

Manually inspect:

* False negatives (missed disease)  
* False positives

Look for:

* Patterns  
* Dataset bias  
* Annotation issues

---

# **21\. Model Optimization**

### **Techniques:**

* Learning rate scheduling  
* Fine-tuning pretrained layers  
* Mixed precision training  
* Gradient clipping

---

# **22\. Final Training (Refinement)**

* Retrain on train \+ validation  
* Use best hyperparameters

---

# **23\. Model Saving**

Save:

* Weights  
* Architecture  
* Preprocessing pipeline

Formats:

* .pt / .pth (PyTorch)  
* .h5 / SavedModel (TensorFlow)

---

# **24\. Deployment Preparation**

* Convert model (ONNX, TensorRT if needed)  
* Optimize latency  
* Batch vs real-time inference

---

# **25\. Inference Pipeline**

Must match training:

* Resize  
* Normalize  
* Same preprocessing

---

# **26\. Monitoring in Production**

Track:

* Input drift  
* Performance decay  
* Data distribution changes

---

# **27\. Retraining Strategy**

* Periodic retraining  
* Add new labeled data  
* Continuous evaluation

---

# **28\. Reproducibility**

Set:

* Random seeds  
* Log:  
  * hyperparameters  
  * dataset version  
  * metrics

Use tools:

* MLflow  
* Weights & Biases

---

# **29\. Documentation**

Document:

* Dataset source  
* Preprocessing  
* Model architecture  
* Evaluation results  
* Limitations

---

# **30\. Ethical & Medical Considerations**

* Bias across populations  
* False negatives risk  
* Not a replacement for doctors  
* Regulatory constraints (FDA, etc.)

---

# **31\. Minimal Example (Pseudo-code)**

### **Training loop idea:**

for epoch in range(num\_epochs):  
   train\_loss \= train(model, train\_loader)  
   val\_loss \= validate(model, val\_loader)

   if val\_loss improves:  
       save\_model()  
---

# **32\. Common Pitfalls (Important)**

* Data leakage (same patient in train/test)  
* Ignoring class imbalance  
* Over-relying on accuracy  
* Not validating preprocessing consistency  
* Training too long without early stopping

---

If you want, I can turn this into:

* a full PyTorch or TensorFlow implementation  
* or a “production-grade” project structure with files and modules  
* or even help you debug your own model step-by-step

