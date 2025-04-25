# Satellite-to-Map-Translation-Robustness-Evaluation

# **1. Project Overview**
This project leverages the Pix2Pix GAN framework to translate satellite images into high-resolution map images. It is designed for applications in urban planning, disaster response, and geospatial analysis. In this updated version, we evaluate the **trustworthiness** of the model under **real-world distortions** to ensure robustness and reliability.

## **2. Objective**

- Build a deep learning model that can translate satellite imagery into accurate map representations.
- Evaluate the **robustness** of the model by testing its performance under common real-world image distortions such as Gaussian noise, blur, and occlusion.
- Use **PSNR** and **SSIM** to quantitatively measure visual fidelity and structural similarity of outputs.

---

## Trustworthiness Focus: Reliability and Robustness

We evaluated the model’s robustness using:

- **Gaussian noise** (σ ≈ 0.1)
- **Gaussian blur** (kernel size: 5×5)
- **Occlusion** (random 30×30 pixel blocks)

The model maintained PSNR > 30 and SSIM > 0.81 across all test conditions, proving resilience to noise and degradation.

**Metrics Used**:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index

---

# **2. Repository Structure**
The repository is structured as follows:

```bash
.
├── README.md
├── Images
│   ├── sample_images
│   └── output_images
└── Code.ipynb
```


---

# **3. Dataset** 
- **Dataset Link: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz** 

The dataset used in this project consists of **satellite images of cities**, obtained via **web scraping from Google Maps**. It is publicly available and provided by **UC Berkeley’s Electrical and Computer Science Department**.

### **1. Dataset Information**
- **Total Images (Train + Validation):** **2,194**
- **Image Format:** Each sample consists of a **concatenated pair of images**, where one half is a **satellite image**, and the other half is a **corresponding map image**.
- **Image Dimensions:**  
  - Each **concatenated image** is **1200 x 600 pixels**  
  - This means each **individual image (either a satellite or map image) is 600 x 600 pixels**
- **Total Dataset Size:** **~257.7 MB**

---

# **4. Environment & Dependencies**
### **Hardware Used**
- **Google Colab with NVIDIA T4 GPU**
- RAM: 16GB
- CUDA enabled for accelerated training

### **Libraries Used**
- **Python 3.x**
- **TensorFlow 2.x**
- **Keras 2.x**
- **Matplotlib**
- **OpenCV**
- **glob, tqdm (for dataset loading & progress tracking)**

---

# **5.  Model Architecture**

## **5.1 Generator: U-Net Architecture for Image-to-Image Translation**

Unlike traditional GANs that map random noise to an image, the **Generator in this project performs image-to-image translation**. This means it **takes an image as input** and transforms it into a different representation of the same image. This task is commonly known as **image translation**, and the framework used here is inspired by **Pix2Pix**.

### **5.1.1. U-Net Architecture**
The **Generator network** follows a **U-Net architecture**, which is a specialized type of **Encoder-Decoder model** with **skip connections** linking the Encoder and Decoder layers. These connections help preserve important spatial information during the transformation process.

In this project, we use **U-Net 256**, which processes **256x256 images** instead of the larger **U-Net 572**. The key difference between the two variants is the input and output image resolution.

### **5.1.2. Functionality**
#### **Encoder (Feature Extraction)**
- The **left side of the U-Net** consists of **convolutional layers** that extract essential features from the input image.
- These extracted features are mapped to a **latent space** (also called a bottleneck representation).
- The latent space is a **1024-dimensional vector** that captures the most significant information about the image.

#### **Decoder (Image Reconstruction)**
- The **right side of the U-Net** consists of **transpose convolutional layers** that map the latent space representation back into a full-sized image.
- The **goal** of the Decoder is **not** to reconstruct the exact input image but to generate a transformed version of it.
- For example, if the input image is a **photo of a dog**, the output image might be a **sketch of the same dog**. The image content remains similar, but the style and representation differ.

### **5.1.3. Role of Skip Connections**
- To **improve training stability** and **retain fine details**, the extracted features from the **Encoder layers** are **directly passed** to corresponding layers in the **Decoder** through **skip connections**.
- This ensures that the Decoder has access to **both high-level features from the latent space** and **low-level spatial details from the Encoder**.
- By doing this, the network efficiently translates images while maintaining their structural integrity.




## **5.2 Discriminator: Convolutional Neural Network for Image Classification**

The **Discriminator network** in this project functions as a **binary classifier** that evaluates whether an input image is **real (from the dataset) or fake (generated by the Generator)**. It follows a **Convolutional Neural Network (CNN) architecture**, optimized for adversarial learning.

### **5.2.1. Architecture Overview**
The **Discriminator is a deep CNN** that progressively **extracts hierarchical features** from an image, learning to **distinguish real images from generated ones**. It processes the input through multiple **convolutional layers**, followed by **fully connected layers**, before producing a **single probability score**.

### **5.2.2. Functionality**
#### **Feature Extraction using Convolutional Layers**
- The Discriminator starts with **several convolutional layers** that scan the image for patterns, edges, and textures.
- Each **Conv layer** applies a **set of filters (kernels)** to extract meaningful representations.
- **Stride and padding** control how much the filter moves across the image.
- **Activation Function:** Uses **LeakyReLU** instead of standard ReLU to prevent **dying neurons** and allow small gradient updates even for negative values.

#### **Downsampling and Spatial Compression**
- After each convolution, the output is **downsampled** using **strided convolutions** (reducing image size and computational cost).
- **Batch Normalization** is applied to stabilize training and speed up convergence.
- **Dropout layers** are introduced to prevent overfitting.

#### **Fully Connected (Dense) Layer & Output**
- The extracted features are **flattened** into a 1D vector.
- This vector is passed through **a dense layer** that aggregates the extracted features.
- **Final Output Layer:** A **single neuron with Sigmoid activation** produces a probability score:
  - **Output close to 1:** The image is classified as **real**.
  - **Output close to 0:** The image is classified as **fake** (generated).

### **5.2.3. Role in Adversarial Training**
- The **Discriminator is trained** using real images from the dataset and fake images generated by the **Generator**.
- The **loss function** used is **binary cross-entropy**, which helps the network **minimize classification error**.
- During training:
  - The **Generator tries to improve** by creating **more realistic images**.
  - The **Discriminator gets better** at identifying **fake images**.
- This adversarial process **pushes both networks to improve**, making the **Generator more effective** in creating realistic images.


---


# **6. Results and Conclusion**

### **6.1. Results**

### Quantitative Evaluation
| Distortion | PSNR (dB) | SSIM |
|------------|-----------|------|
| Original   | 31.10     | 0.8356 |
| Noisy      | 30.57     | 0.8191 |
| Blurred    | 30.76     | 0.8254 |
| Occluded   | 30.74     | 0.8292 |

### Visual Output
![Visual Grid](outputs/visual_grid_example.png)

---

### **6.2. Conclusion**
- The **U-Net-based Generator** successfully learned to translate satellite images into maps.
- The **use of skip connections in U-Net** helped retain fine details in the output images.
- The **CNN-based Discriminator** provided effective feedback, ensuring the Generator improved over time.
- **Adversarial training was effective**, with both networks evolving in response to each other.
- The **quality of generated maps improved significantly after ~100 epochs**, producing results close to the real dataset.

### **6.3. Future Improvements**
Although the model performed well, there are some areas for further enhancement:
- **Train on higher-resolution images** (e.g., 1024x1024) for more detailed outputs.
- **Use a more advanced Discriminator**, such as **PatchGAN**, which focuses on local features rather than the entire image.
- **Experiment with different loss functions**, such as **L1 loss** or **perceptual loss**, to improve visual accuracy.
- **Augment the dataset** with more diverse satellite imagery to improve generalization.

The results demonstrate that **image-to-image translation using GANs can effectively transform satellite images into maps**, with potential applications in **geospatial analysis, urban planning, and automated cartography**.

---
