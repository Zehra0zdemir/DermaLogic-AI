# DermaLogic AI V3: Multimodal Hybrid Skincare Diagnosis & Prescribing Engine

DermaLogic AI V3 is a production-grade **Multimodal Hybrid Deep Learning Pipeline** developed using the **Keras Functional API**. The system concurrently processes heterogeneous data streams—combining **spatial facial images** and **tabular clinical/survey metadata**—to classify dominant skin conditions and algorithmically compute personalized, 100% certified vegan active ingredient concentration percentages.

##  Key Engineering Milestones & Metrics
* **Multimodal Architecture:** Successfully fused high-dimensional latent visual features with structured medical metadata vectors into a single 80-dimensional representation space.
* **Overfitting Elimination Delta (+30.0%):** Integrated an on-the-fly spatial RAM data augmentation generator (`ImageDataGenerator`) and a 50% Dropout regularizer, completely breaking a severe 42.0% baseline overfitting bottleneck to stabilize validation accuracy at a reliable **72.0%**.
* **Lightweight Optimization:** Scaled input tensors to a native $224 \times 224 \times 3$ footprint, matching pre-trained weights perfectly and compressing training runtimes to **under 10 seconds per epoch** on standard cloud-hosted T4 GPUs.

##  System Architecture & Model Topology

The pipeline utilizes a non-linear Directed Acyclic Graph (DAG) structure to merge diverse data inputs smoothly:
[Visual Input: 224x224x3 Matrix] ----> [MobileNetV2 Core (Frozen)] ----> [64-Dim Embedding Vector] ---

|--> [Concatenate Fusion Layer] --> [Dense 64 + Dropout 0.5] --> [3-Class Softmax Output]
[Tabular Input: 15-Dim Vector]   ----> [Dense 32 -> Dense 16 layers] ----> [16-Dim Latent Vector] ----/

### 1. Visual Convolutional Branch (CNN)
* Core Feature Extractor: `MobileNetV2` (Pre-trained on ImageNet, weights frozen via `trainable=False`).
* Leverages **Depthwise Separable Convolutions** to minimize parameter overhead while extracting abstract structural skin markers.
* Flattened via `GlobalAveragePooling2D` and mapped to a 64-dimensional dense latent embedding.

### 2. Tabular Clinical Branch (ANN)
* Processes a 15-dimensional fused feature vector ($10\text{ Spreadsheet Features} + 5\text{ Presentation Survey Features}$ including Patient Age, Skin Type, and Tolerance).
* Implements a Multi-Layer Perceptron (MLP) block (`Dense 32 -> Dense 16`) using `ReLU` activations to isolate non-linear cross-feature correlations.

##  Loss Optimization & Prescribing Logic

The model compiles via the **Adam Optimizer** (Learning Rate = $0.001$) and minimizes error using **Categorical Crossentropy Loss** across 3 discrete diagnostic classes:

| Diagnostic Target Output Class | Dominant Clinical Profile Indicated |
| :--- | :--- |
| **Class 0** | Active Acne State (Severe Acne, Blackheads, Whiteheads Focus) |
| **Class 1** | Vascular Inflammation & Sensitivity (High Irritation/Redness Focus) |
| **Class 2** | Structural Aging & Hyperpigmentation (Wrinkles & Dark Spots Focus) |

The final Softmax layer yields an actionable probability distribution mapping. An algorithmic, rule-based expert decision head intercepts this neural diagnosis alongside real-time user constraints to immediately output strict prescriptive chemical formulation yields (e.g., *%2 Salicylic Acid (BHA)* for active sebum control or *Centella Asiatica (Cica)* for hyper-sensitive epidermal barrier repair).

##  Tech Stack & Libraries
* **Core Frameworks:** Python, TensorFlow, Keras (Functional API), Pandas, NumPy
* **Visualization:** Matplotlib (Comprehensive 3-Way Overfitting & Convergence Evaluation Subplots)
* **Pre-trained Core:** MobileNetV2 Architecture
