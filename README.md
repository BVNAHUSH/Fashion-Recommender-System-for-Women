
# 🧥 Fashion Image Recommendation System using CNN (VGG16)

This project builds a **Content-Based Fashion Recommendation System** using deep learning. By extracting visual features from fashion images using a pretrained **VGG16 CNN**, it recommends top similar outfits based on cosine similarity.

---

## 📂 Project Overview

This project implements a **Content-Based Image Recommendation System tailored for women's fashion**, utilizing deep learning techniques to identify and suggest visually similar clothing items.

The system is built using **VGG16**, a well-established convolutional neural network pretrained on ImageNet, to extract deep visual features from product images. These features are then compared using cosine similarity to find and recommend items that closely match the input image in terms of **style, color, and texture**.

Key highlights include:

* Automatic feature extraction without manual tagging or labeling.

* End-to-end image handling, from data extraction (via Google Drive) to similarity computation.

* Clean visualization of recommendations for intuitive interpretation.

* Scalable architecture adaptable to larger datasets or real-time applications.

This project showcases how computer vision and transfer learning can be applied in fashion-tech, enabling smart recommendation systems for online retail, personal styling apps, or virtual storefronts.

---
---

🔧 **Features**
* 👗 Fashion image processing using PIL and Matplotlib

* 🧠 Feature extraction using VGG16 pretrained on ImageNet

* 🔍 Content-based recommendation using cosine similarity

* 📦 Dataset handling via Google Drive and zipfile

* 📊 Visual display of input & similar items using Matplotlib
---

## 🔧 Setup & Requirements

This code is designed to run in **Google Colab**.

### Install Dependencies (Colab usually has them pre-installed)

```python
!pip install tensorflow pillow matplotlib
```

---

## 🚀 How It Works

### 1. **Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

Mounts Google Drive to access the dataset.

---

### 2. **Extract ZIP File**

Extracts the dataset (`women-fashion.zip`) to a local directory:

```python
zip_file_path = '/content/drive/MyDrive/women-fashion.zip'
extraction_directory = '/content/women_fashion/'
```

---

### 3. **Display Images**

Using PIL and Matplotlib to visualize images:

```python
from PIL import Image
import matplotlib.pyplot as plt

def display_image(file_path):
    image = Image.open(file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
```


 ![image ](https://github.com/user-attachments/assets/2f18af3d-e26b-4972-85d2-22c07e602d99)

---

### 4. **Preprocess Images**

All valid image formats (`.jpg`, `.png`, `.jpeg`, `.webp`) are loaded and preprocessed to match VGG16 input requirements:

```python
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)
```

---

### 5. **Feature Extraction**

Features are extracted using VGG16’s convolutional layers and normalized:

```python
def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features
```

---

### 6. **Similarity Comparison**

Cosine similarity is used to compare the input image to all dataset images:

```python
def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]
    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]
```

---

## 🖼 Sample Recommendation Output

The input image and top-N similar fashion items are displayed side-by-side using Matplotlib.

![image](https://github.com/user-attachments/assets/93c90d07-a49c-430c-bc3b-2e90d46a3096)

---

## 📌 Example Usage

```python
input_image_path = '/content/women_fashion/women fashion/dark, elegant, sleeveless dress that reaches down to about mid-calf.jpg'
recommend_fashion_items_cnn(input_image_path, all_features, image_paths_list, model, top_n=4)
```

---

## 📁 Directory Structure (Post-Extraction)

```
/content/
├── drive/
│   └── MyDrive/
│       └── women-fashion.zip
├── women_fashion/
│   └── women fashion/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
```

---

## 📊 Technologies Used

* Python
* TensorFlow & Keras
* VGG16 CNN model
* PIL, Matplotlib
* NumPy, SciPy

---

## 🛠 Future Improvements

* Add a **web UI** using **Streamlit** or **Flask**
* Replace VGG16 with more powerful models like **ResNet**, **EfficientNet**, or **CLIP**
* Integrate **text-based search** for multimodal fashion recommendations
* Implement **clustering (e.g., KMeans)** or **FAISS** for faster similarity search

---

## 📚 References

* [Keras Applications: VGG16](https://keras.io/api/applications/vgg/)
* [Cosine Similarity – Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
* [TensorFlow Image Preprocessing](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image)

---

## 📄 License

This project is licensed under the **MIT License**.


---

## 👤 About Me

**Developed by:** *B V Nahush*
🎓 Final-year AI & ML Engineering Student

💡 Passionate about Deep Learning, Computer Vision & Recommender Systems

📫 Reach me here:

* 🔗 [LinkedIn](https://www.linkedin.com/in/b-v-nahush?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
* 💻 [GitHub](https://github.com/nahush919)
* 📧 Email: [work.nahushreddy@gmail.com](mailto:work.nahushreddy@gmail.com)

---




