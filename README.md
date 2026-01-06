# Traffic Sign Recognition using Deep Learning

As research continues in the development of self-driving cars, one of the most important challenges is enabling machines to understand their surroundings through computer vision. A critical part of this process is the ability to accurately recognize and distinguish road signs such as stop signs, speed limit signs, yield signs, and more.

In this project, a neural network is built using **TensorFlow** to classify traffic signs from images. The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, which contains thousands of labeled images across **43 different traffic sign categories**. This project is part of **CS50’s Introduction to Artificial Intelligence with Python**.

---

## 📊 Dataset

The GTSRB dataset is organized into 43 subdirectories, numbered from `0` to `42`, where each directory represents a unique type of traffic sign. Each directory contains multiple images of the corresponding sign captured under varying lighting, angles, and resolutions. This diversity makes the dataset suitable for training a robust image classification model.

---

## 🧠 Approach & Model Design

The project uses a **Convolutional Neural Network (CNN)** to learn spatial patterns in traffic sign images. Images are first loaded and resized to a fixed dimension to ensure consistency before being passed into the neural network.

During experimentation, several architectural choices were explored:
- Convolutional layers with different filter sizes to extract visual features
- Max-pooling layers to reduce dimensionality
- Fully connected (dense) layers to perform classification
- Dropout layers to reduce overfitting

Shallower models trained quickly but failed to generalize well, while deeper models with additional convolutional layers achieved better accuracy but required more training time. Adding dropout improved validation performance by preventing the model from memorizing the training data.

---

## 🧪 Training & Evaluation

The dataset is split into training and testing sets. The model is trained on the training data and evaluated on unseen test data to measure accuracy. After training, the model demonstrates strong performance in identifying traffic sign categories, highlighting the effectiveness of CNNs in computer vision tasks.

---

## 🛠️ Technologies Used

- **Python 3**
- **TensorFlow / Keras**
- **OpenCV (cv2)** for image processing
- **NumPy**
- **scikit-learn**

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:
- Image preprocessing using OpenCV
- Building and training CNNs with TensorFlow
- Working with real-world computer vision datasets
- Experimenting with neural network architectures and hyperparameters

---

B.Tech CSE (Artificial Intelligence)  
CS50 AI Project
