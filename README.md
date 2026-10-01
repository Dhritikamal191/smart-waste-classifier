# Smart Waste Classifier



An AI-powered waste classification system built using deep learning,

EfficientNetB0, TensorFlow, FastAPI, Docker, and NVIDIA GPU acceleration.



## Overview



Smart Waste Classifier automatically identifies waste from an uploaded

image and predicts its waste category using a fine-tuned EfficientNetB0

image classification model.


## Supported Classes



- Battery

- Biological

- Brown Glass

- Cardboard

- Clothes

- Green Glass

- Metal

- Paper

- Plastic

- Shoes

- Trash

- White Glass



## Dataset



- Images: 15,515

- Classes: 12

- Training images: 10,860

- Validation images: 2,327

- Test images: 2,328



The dataset was processed using duplicate-aware splitting to reduce

data leakage between training, validation, and test sets.



## Model



Architecture:



EfficientNetB0



Training approach:



1. Transfer learning

2. Classification head training

3. Fine-tuning

4. Evaluation

5. Robustness testing

6. Grad-CAM explainability


## Model Artifact



```text

models/waste\_classifier\_finetuned.keras

