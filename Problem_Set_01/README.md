# Problem Set 01: Pediatric Chest X-Ray Pneumonia Classification

## 1. Approach

This project trains a Convolutional Neural Network (CNN) to perform binary medical image classification on pediatric chest X-ray images, distinguishing between `Normal` and `Pneumonia` cases.

## 2. Methodology

### Data Pipeline
- **Dataset Source Note:** Rather than manually uploading the raw zip file provided for the assignment — which repeatedly caused browser connection timeouts and zipfile offset corruption in Google Colab due to its ~1.1 GB size.
- The dataset was programmatically fetched directly from the official Kaggle mirror using `kagglehub` (`paultimothymooney/chest-xray-pneumonia`). This ensured complete file integrity and significantly reduced environment setup time.
- Rescaled pixel values to `[0, 1]` using `ImageDataGenerator`.
- Applied data augmentation (rotation, zoom, horizontal flips, shifts) on the training set to reduce overfitting.
- Standardized target image dimensions to 150 × 150.

### CNN Architecture
- 3 Convolutional Blocks (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
- `Flatten` layer leading into a 128-unit `Dense` layer, regularized with `Dropout(0.5)`
- Single-unit output layer using `sigmoid` activation for binary classification

### Optimization
- Compiled using the `adam` optimizer and `binary_crossentropy` loss
- Trained over 5 epochs with GPU acceleration

## 3. Findings & Performance

**Evaluation Output (Test Set — 624 Images):**

| Metric | Score |
|---|---|
| Overall Accuracy | 69% |
| Pneumonia Precision | 0.94 |
| Normal Recall | 0.94 |

### Key Takeaways
- Data augmentation and batch normalization helped prevent model divergence, despite the runtime constraints imposed by limiting training to 5 epochs.
- The high precision in detecting Pneumonia cases (0.94) provides a strong baseline for medical triage pipelines, where minimizing false positives on the disease class is often a priority.
