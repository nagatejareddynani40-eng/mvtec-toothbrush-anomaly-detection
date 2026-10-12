# mvtec-toothbrush-anomaly-detection
Deep learning-based image anomaly detection using a convolutional autoencoder on the MVTec AD toothbrush dataset.
# Deep Learning-Based Image Anomaly Detection Using Convolutional Autoencoders

## Project Overview

This project implements a **Convolutional Autoencoder (CAE)** for image anomaly detection using the toothbrush category of the MVTec Anomaly Detection (MVTec AD) dataset.

The objective is to identify defective toothbrush images by learning the visual patterns of normal images. The trained model reconstructs input images, and reconstruction errors are used to identify potential anomalies.

This project demonstrates the application of deep learning and computer vision to industrial quality inspection.

## Objectives

- Understand image-based industrial anomaly detection.
- Build and train a convolutional autoencoder using deep learning.
- Reconstruct input images and calculate reconstruction errors.
- Detect potential defects using an anomaly threshold.
- Visualize original images, reconstructed images, and anomaly heatmaps.
- Evaluate the model using test-set classification metrics.

## Dataset

**Dataset:** MVTec Anomaly Detection (MVTec AD)

**Category:** Toothbrush

**Official dataset source:** https://www.mvtec.com/company/research/datasets/mvtec-ad

The dataset contains industrial images of normal products and defective products. It is designed for research into unsupervised anomaly detection.

The dataset is not included in this repository because of its size.

### Dataset Setup

1. Download the MVTec AD dataset from the official source.
2. Extract the dataset on your local computer.
3. Locate the `toothbrush` category folder.
4. Update the dataset path in the notebook to match your local directory.
5. Run the notebook cells in order.

Example local dataset path:

```text
mvtec_anomaly_detection/
└── toothbrush/
    ├── ground_truth/
    ├── test/
    └── train/
```

The exact folders used by the notebook should match the dataset structure expected by the code.

## Methodology

### 1. Data Preparation
- Load the toothbrush images.
- Preprocess and resize images to the required input dimensions.
- Convert images into tensors and normalize pixel values.
- Prepare the training and testing data.

### 2. Convolutional Autoencoder
The model consists of two main components:

- **Encoder:** Extracts important visual features and compresses the input image into a compact representation.
- **Decoder:** Reconstructs the image from the encoded representation.

The model learns to reconstruct normal image patterns. Defects may produce larger reconstruction errors.

### 3. Anomaly Detection
- Pass test images through the trained autoencoder.
- Compare each input image with its reconstruction.
- Calculate reconstruction errors.
- Apply a selected anomaly threshold to classify images as normal or anomalous.
- Generate anomaly heatmaps to visualize regions with high reconstruction error.

### 4. Evaluation
Evaluate the model by comparing its predictions against the available test labels.

## Technologies Used

- Python
- Jupyter Notebook
- PyTorch
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- MVTec AD dataset

## Experimental Results

The following results were obtained during the project's final test evaluation.

| Metric | Result |
|---|---:|
| Test images | 26 |
| Classification accuracy | 76.92% |
| Anomaly threshold | 0.096158 |

The threshold was used to classify test images based on their anomaly scores.

**Important:** These results represent the current experimental run and should not be interpreted as proof of reliable industrial deployment.

## Visualizations

The project includes visual analysis of:

- Original toothbrush images
- Autoencoder-reconstructed images
- Anomaly heatmaps
- Reconstruction errors and predicted anomaly labels

Selected result images can be added to a `results/` folder and displayed here.

Example:

```markdown
![Original Image](results/original_image.jpg)

![Reconstructed Image](results/reconstructed_image.jpg)

![Anomaly Heatmap](results/anomaly_heatmap.jpg)
```

Replace these example filenames with the actual names of your saved images.

## Project Structure

```text
mvtec-toothbrush-anomaly-detection/
├── README.md
├── toothbrush_anomaly_detection.ipynb
├── requirements.txt
└── results/
    ├── original_image.jpg
    ├── reconstructed_image.jpg
    └── anomaly_heatmap.jpg
```

The `results/` images shown above are examples of suggested filenames. Add the files only after saving the corresponding outputs.

## Limitations

- The final test evaluation used only 26 images, so the measured accuracy may not generalize to other images.
- Reconstruction errors do not always correspond to actual defects.
- The selected anomaly threshold can affect false-positive and false-negative predictions.
- The model has been evaluated on the toothbrush category, not necessarily on all MVTec AD categories.
- Additional evaluation, threshold validation, and testing on more representative data are required before considering industrial use.

## Future Improvements

- Evaluate on a larger and more representative test set.
- Tune the anomaly threshold using validation data.
- Report precision, recall, F1-score, ROC-AUC, and image-level or pixel-level metrics where applicable.
- Compare the convolutional autoencoder with other anomaly detection methods.
- Improve anomaly localization and heatmap quality.
- Develop a simple interface for uploading an image and viewing its anomaly score.

## How to Run

1. Clone or download this repository.
2. Install the required Python packages.
3. Download and extract the MVTec AD dataset separately.
4. Update the dataset path in the notebook.
5. Open `toothbrush_anomaly_detection.ipynb` in Jupyter Notebook.
6. Run the cells in sequence.

## Author

Developed as a deep learning and computer vision project focused on industrial image anomaly detection.

## Acknowledgment

The project uses the MVTec Anomaly Detection dataset for research and experimentation. Please refer to the official dataset source for its documentation and usage terms.
