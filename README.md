# 3-Layer-Deep-Neural-Network-for-Non-Linear-Regression

## Overview
This repository contains multiple implementations of a **3-layer deep neural network** for **non-linear regression** using **NumPy, PyTorch, and TensorFlow**. Each implementation explores different approaches, ranging from scratch implementations to high-level API usage.

### Key Features:
- **TensorFlow Einsum Usage**: TensorFlow implementations use `einsum` instead of traditional matrix multiplication.
- **3-Variable-Based Non-Linear Equation**: The dataset is generated based on a **three-variable equation**, unlike previous two-variable examples.
- **Synthetic Data Generation & 4D Plotting**: Data is generated and visualized in **4D**.
- **Full GitHub Documentation**: Includes **Colab links**, code explanations, and a **video walkthrough** for each implementation.
- **Detailed Video Walkthroughs**: Executed **Colab notebooks** are documented with step-by-step **code explanations**.

---

## 📊 Data Generation

A **synthetic dataset** is generated using the following **3-variable-based non-linear equation**:

\[ y = \sin(x_1) + \cos(x_2) + x_3^2 \]

where **x₁, x₂, and x₃** are randomly sampled values between -1 and 1.

This dataset is used across all implementations for training and evaluation.

---

## 🚀 Implementations

### 1️⃣ NumPy-Only Implementation (Manual Backpropagation)

**File:** [colab_numpy_scratch.ipynb](https://colab.research.google.com/github/roshini-joga/3-Layer-Deep-Neural-Network-for-Non-Linear-Regression/blob/master/NumPy(3_layer).ipynb)

🔹 Implements a **3-layer deep neural network from scratch** using **NumPy**.
🔹 Uses **non-linear activation functions**.
🔹 Implements **manual backpropagation** using the **chain rule**.
🔹 Displays **loss progression, training epochs, and final output**.

---

### 2️⃣ PyTorch Implementations

#### 🔹 PyTorch Without Built-in Layers
**File:** [colab_pytorch_scratch.ipynb](./colab_pytorch_scratch.ipynb)

- **Implements the neural network manually** (no `nn.Module`, no built-in layers).
- Uses **manual backpropagation**.
- **Weight updates are handled manually** using gradients.

#### 🔹 PyTorch Class-Based Model
**File:** [colab_pytorch_class.ipynb](./colab_pytorch_class.ipynb)

- Uses `nn.Module` and `nn.Linear` layers.
- Implements **forward propagation** and **automatic backpropagation**.
- Uses PyTorch's built-in **Adam optimizer**.

#### 🔹 PyTorch Lightning Implementation
**File:** [colab_pytorch_lightning.ipynb](./colab_pytorch_lightning.ipynb)

- Uses **PyTorch Lightning** to manage training.
- Implements the model as a **LightningModule**.
- Uses **`Trainer`** for efficient training.

---

### 3️⃣ TensorFlow Implementations

#### 🔹 TensorFlow Without High-Level APIs
**File:** [colab_tf_lowlevel.ipynb](./colab_tf_lowlevel.ipynb)

- Implements a **3-layer neural network manually** without using high-level TensorFlow functions.
- Uses **TensorFlow’s lower-level tensor operations**.

#### 🔹 TensorFlow Built-in Layers
**File:** [colab_tf_builtin_layers.ipynb](./colab_tf_builtin_layers.ipynb)

- Uses **built-in layers** from TensorFlow (`tf.keras.layers.Dense`).
- Implements **automatic backpropagation and weight updates**.

#### 🔹 TensorFlow Functional API
**File:** [colab_tf_functional_api.ipynb](./colab_tf_functional_api.ipynb)

- Implements the model using **TensorFlow’s Functional API**.
- More flexible than sequential models, allowing complex architectures.

#### 🔹 TensorFlow High-Level API
**File:** [colab_tf_highlevel_api.ipynb](./colab_tf_highlevel_api.ipynb)

- Uses **high-level APIs** such as `tf.keras.Sequential`.
- Simplifies model definition and training.

---

### 4️⃣ TensorFlow Einsum Requirement
All TensorFlow implementations replace **matrix multiplication** with **`einsum`** to optimize computations.

Example:
```python
# Instead of: output = tf.matmul(X, W)
output = tf.einsum('ij,jk->ik', X, W)
```

---

### 5️⃣ Data Visualization (4D Plot)

- The **synthetic dataset** is **visualized using a 4D plot**.
- This helps analyze how data is distributed across multiple dimensions.
- **File:** `data_generation.py`

---
## 📽️ Video Walkthroughs
Each implementation has an accompanying **video walkthrough**, explaining the code structure and execution.
- **[Watch NumPy Scratch Implementation Walkthrough](#)**
- **[Watch PyTorch Scratch Implementation Walkthrough](#)**
- **[Watch PyTorch Class-Based Walkthrough](#)**
- **[Watch PyTorch Lightning Walkthrough](#)**
- **[Watch TensorFlow Implementations Walkthrough](#)**

(The links will be updated once the videos are uploaded.)

---

## 📝 To-Do
✅ Implement TensorFlow versions using Einsum.
✅ Include functional API and high-level API versions.
✅ Upload Colab links and execution results.
✅ Add additional experiments and insights.

---

## 🔗 References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Google Colab](https://colab.research.google.com/)

---

**Author:** Roshini Joga 🚀

For any questions, feel free to reach out! 🎯

