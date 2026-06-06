# 🖼️ Image Caption Generator

> *"Automatically generating descriptive textual captions for images using deep learning."*

![Model](https://img.shields.io/badge/Model-CNN_%2B_LSTM-blue)
![Backend](https://img.shields.io/badge/Backend-Flask-lightgrey)
![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS-orange)

## 📖 Overview
This project is a machine learning web application designed to automatically generate descriptive text captions for uploaded images. By combining Convolutional Neural Networks (CNN) for image analysis with sequence-generating neural networks, the system "looks" at an image and translates its visual features into human-readable text.

---

## 🧠 Core Features & Mechanics

### 👁️ 1. Deep Learning Architecture
* **Visual Feature Extraction:** Utilizes a pre-trained **DenseNet201** model to extract rich, 1920-dimensional visual feature vectors from uploaded images.
* **Sequence Generation:** Employs a custom-trained Keras captioning model alongside a tokenizer to predict and assemble the caption word-by-word.
* **Greedy Decoding:** The text generation relies on a greedy search approach, predicting the most probable next word in the sequence until an end token is reached.

### 🌐 2. Interactive Web Interface
* **Live Image Preview:** Features a clean, responsive frontend built with modern CSS and Vanilla JavaScript that provides an instant, client-side preview of the image before it is processed.
* **Seamless Integration:** Built on a lightweight **Flask** backend that seamlessly bridges the user-facing web interface with the heavy TensorFlow/Keras inference logic.
* **Responsive Design:** Utilizes a clean, card-based UI with dynamic gradients and the 'Inter' font family for a polished user experience.

---

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3, Vanilla JavaScript 
* **Backend:** Python, Flask
* **Machine Learning:** TensorFlow, Keras (DenseNet201)
* **Data Processing & Utilities:** NumPy, Pillow (PIL), Pickle
