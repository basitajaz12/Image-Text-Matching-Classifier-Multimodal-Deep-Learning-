# Image-Text-Matching-Classifier-Multimodal-Deep-Learning-
This project implements a multimodal Image–Text Matching (ITM) classifier that determines whether a given image and text caption match or do not match.
This project implements a multimodal Image–Text Matching (ITM) classifier that determines whether a given image and text caption match or do not match.
The system combines visual features learned from images and semantic features extracted from text embeddings, enabling robust cross-modal understanding.

The classifier is trained and evaluated on the Flickr8k dataset, using a CNN-based image encoder and precomputed sentence embeddings for text.

🎯 Key Features

Binary image–text matching (match / no-match)

Multimodal learning with image + text inputs

CNN-based image feature extraction

Pretrained sentence embeddings for text representation

Efficient training using TensorFlow data pipelines

Evaluation with accuracy and prediction inspection

🧠 Task Description

Given:

🖼️ an image

📝 a text caption

The model predicts:

Match → the caption correctly describes the image

No-Match → the caption does not describe the image

This task is fundamental in applications such as:

Image retrieval

Visual question answering

Cross-modal search

Multimodal AI systems

📊 Dataset

Flickr8k Dataset

Real-world images with multiple captions

Split into training, validation, and test sets

Captions converted into dense sentence embeddings (384-dimensional)

⚙️ System Architecture
🔹 Image Encoder

Convolutional Neural Network (CNN)

Extracts visual features directly from raw image pixels

Multiple Conv2D + MaxPooling layers

Flattened and projected into a shared embedding space

🔹 Text Encoder

Uses precomputed sentence embeddings

Dense projection layers align text features with image features

Efficient since embeddings are loaded once and reused

🔹 Multimodal Fusion

Image and text embeddings are concatenated

Passed through dense layers

Final softmax classifier predicts match / no-match

🏗️ Model Pipeline

Load image and sentence embedding

Extract visual features using CNN

Project both image and text features

Concatenate multimodal features

Classify using a dense neural network

🛠️ Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Einops

Sentence Transformers (precomputed embeddings)


🚀 How to Run the Project
1️⃣ Install Dependencies
pip install tensorflow tf-models-official tensorflow-text einops matplotlib
2️⃣ Update Dataset Path

In ITM_DataLoader, set:

IMAGES_PATH = "path/to/flickr8k-resised"
3️⃣ Run the Classifier
python ITM_Classifier.py

⚠️ GPU recommended — running on CPU will be very slow.

📈 Training & Evaluation

Optimizer: AdamW

Loss Function: KL Divergence

Metric: Binary Accuracy

Training includes validation monitoring

Evaluation performed on unseen test data

The system reports:

Sample predictions

Overall test accuracy

TensorFlow-computed evaluation metrics

🧪 Output Example
Caption: "A dog running through the grass"
Prediction: Match
Image: flickr_12345.jpg
🔮 Future Improvements

Replace CNN with pretrained vision models (ResNet, EfficientNet)

End-to-end text encoder instead of fixed embeddings

Attention-based multimodal fusion

Web interface for interactive testing

Support for larger datasets (MS-COCO)
