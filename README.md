# Multitask Multi attention Residual Shrinkage Convolutional Neural Network for Content-based Image Retrieval using cosine similarity and Attention feature fusion of AlexNet, VGG19 and ResNet50

This repository implements a **Content-Based Image Retrieval (CBIR)** system using a custom deep learning model named **MARSCNN** along with pre-trained models like **AlexNet, VGG19, and ResNet50** for feature extraction. The extracted features are concatenated using an **attention-based feature fusion technique** to enhance retrieval accuracy. The performance of the models is compared using key metrics.

## Overview

This project leverages **Google Colab** for training and evaluating the model, utilizing the available GPU resources. It uses **MARSCNN (Multitask Multi Attention Residual Shrinkage CNN)** for feature extraction, combined with **AlexNet, VGG19, and ResNet50**, and fuses their features using an attention mechanism for efficient image retrieval.

### Key Features:
- **Custom MARSCNN architecture** with attention and residual shrinkage layers.
- **Attention-based feature fusion** to combine the strengths of **AlexNet, VGG19, and ResNet50**.
- Model performance is evaluated using metrics such as **Accuracy, Precision, Recall, F1 Score, and Error Rate**.
- Seamless integration with **Google Colab**, supporting GPU acceleration.

## Model Performance Comparison

We compared the performance of the individual models (AlexNet, VGG19, ResNet50) and the combined model (MARSCNN with attention-based feature fusion) using the following metrics:

| Accuracy  | Precision | Recall   | F1 Score | Error Rate | Model            |
|-----------|-----------|----------|----------|------------|------------------|
| 0.878840  | 0.878840  | 0.878840 | 0.878840 | 0.121160   | MARSCNN_AlexNet   |
| 0.974403  | 0.966302  | 0.956075 | 0.959300 | 0.025597   | MARSCNN_VGG19     |
| 0.931741  | 0.931741  | 0.931741 | 0.931741 | 0.068259   | MARSCNN_ResNet50  |
| 0.972696  | 0.972696  | 0.972696 | 0.972696 | 0.027304   | MARSCNN_combined  |

### Key Observations:
- **MARSCNN_VGG19** achieved the highest **precision** and **recall**, but **MARSCNN_combined** performed best in terms of **F1 Score** and overall **accuracy**.
- The combined model shows a significant improvement due to the attention-based feature fusion, which leverages the strengths of all models.

## Prerequisites

Before running the notebook in **Google Colab**, ensure the following setup:

1. **Open the Notebook in Colab:**
   - Upload the Jupyter notebook (`MARSCNN_CBIR_CS_AN_VGG19_RN50.ipynb`) to Google Colab or link your GitHub repository directly.

2. **Mount Google Drive (Optional):**
   If your dataset is stored in Google Drive, mount it by running:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Install Required Libraries:**

   Add the following to install dependencies:

   ```python
   !pip install torch torchvision matplotlib numpy opencv-python
   ```

## Dataset

This project uses the **Caltech 101 dataset** for training and testing. Download the dataset or load it from Google Drive, and organize it as follows:

```
/content/drive/MyDrive/Caltech101
    /class1
        image1.jpg
        image2.jpg
        ...
    /class2
        image1.jpg
        image2.jpg
        ...
```

To download directly in Colab:

```python
!wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
!tar -xvzf 101_ObjectCategories.tar.gz
```

## Running the Project in Google Colab

1. **Clone the Repository or Upload the Notebook:**

   Clone the repository or upload the notebook to Colab:

   ```bash
   !git clone https://github.com/yourusername/MARSCNN_CBIR.git
   ```

2. **Enable GPU Support:**
   Ensure GPU is enabled by navigating to:
   `Runtime` > `Change runtime type` > `Hardware Accelerator` > `GPU`.

3. **Run the Notebook:**
   Execute the cells to preprocess data, extract features using **MARSCNN**, **AlexNet**, **VGG19**, and **ResNet50**, and fuse them using the attention mechanism. Then, retrieve the most similar images.

4. **Save and Display Results:**
   The CBIR system will return the most similar images along with their similarity scores.

## Model Architecture and Feature Fusion

- **MARSCNN**: A custom CNN with attention and residual shrinkage mechanisms.
- **AlexNet, VGG19, ResNet50**: Pre-trained models for feature extraction.
- **Attention-based feature fusion**: Combines the feature vectors from the pre-trained models for improved retrieval accuracy.

## Example of Use in Colab

```python
# Example to retrieve similar images
query_image = 'path/to/query/image.jpg'
retrieve_similar_images(query_image)
```

## Future Enhancements

- Explore different similarity measures to further improve accuracy.
- Experiment with larger datasets for scalability.
- Implement advanced preprocessing techniques for more robust image retrieval.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
