# 🌱 Weed Species Classification: CBIR Laser Weeder & MFWD Dataset

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)
![NMSLIB](https://img.shields.io/badge/NMSLIB-HNSW-red.svg)

This repository serves a dual purpose. It is a fork of the **Moving Fields Weed Dataset (MFWD)** tools, extended with a custom **Content-Based Image Retrieval (CBIR)** system designed to simulate the vision engine of a **"Laser Weeder"**. The system accurately differentiates between commercial crops and invasive weeds at early growth stages using a highly optimized, few-shot feature extraction architecture.

---

## 🚀 Part 1: CBIR Laser Weeder Engine

Instead of relying solely on massive Deep Learning classifiers that require extensive retraining for every new weed species, our project implements a **Hybrid CBIR System**. By comparing the visual features of a target plant against a pre-indexed database, the system can generalize from very few examples (Few-Shot), calculating a distance threshold to trigger a weeding laser.

### 🧠 Hybrid Feature Architecture
Our system extracts a concatenated feature vector of **2526 dimensions** per image:
1. **Deep Semantic Features (CNN):** `ResNet50` (Global Average Pooling) -> 2048 dims
2. **Structural Shape (HOG):** Histogram of Oriented Gradients -> 324 dims
3. **Local Texture (LBP):** Uniform Local Binary Patterns (Radius 3) -> 26 dims
4. **Color Context (HSV):** 3-channel Histogram -> 96 dims
5. **Keypoints (ORB):** Average pooled descriptors -> 32 dims *(Tested and documented)*

To meet the real-time requirements of an agricultural vehicle, we implemented **NMSLIB** to build a **Hierarchical Navigable Small World (HNSW)** graph, reducing query latency to $O(\log N)$ using the L2 (Euclidean) distance metric.

### 💻 Usage of the CBIR System
Run the main script to initialize the CBIR engine. It will automatically extract features and build the HNSW graph if no index is found in the local disk.

```bash
python cbir_system.py
```
- `cbir.index_folder(folder_path)`: Extracts hybrid features and builds the NMSLIB graph.
- `cbir.search(query_path, top_k=5)`: Returns the $K$ most similar plants.
- `cbir.visualize_raw_stats(query, results)`: Plots distance breakdowns for each descriptor to visualize hyperparameter impacts.

---

## 🌿 Part 2: Moving Fields Weed Dataset (MFWD)

This repository also contains the original code to download and process the **Moving Fields Weed Dataset (MFWD)**, a manually annotated and curated dataset of diverse weed species in maize and sorghum.

### 📥 Download Data
Use the following code to download specific parts of the dataset via FTP:

a) All files annotated with segmentation masks:
```bash
python3 download_by_ftp.py masks
```
b) All images of multiple specific species (e.g., ACHMI, CHEAL):
```bash
python3 download_by_ftp.py species 'ACHMI, CHEAL'
```
c) All images of multiple specific trays:
```bash
python3 download_by_ftp.py trays '109801, 109802'
```
**d) (New implementation) Downloads a fixed number (in the code) of random species at their last stage:**
```bash
python3 download_by_ftp.py variety
```
*Possible arguments:* `save_path`, `files` (EPPO codes or tray IDs), `img_type` ('jpegs' or 'pngs').

### ⚙️ Dataset Utilities & Baseline
- **Re-scale Data:** Use `rescale_data.py` to change the resolution of the downloaded images.
- **Model Baseline:** 1) Run `prepare_data.py` to generate images of the plant cut-outs.
  2) Run `optimize_hyperparameters_efficientnet.py` to reproduce the EfficientNet_b0 classification baseline.
  3) Run `test.py` to generate the weighted f1-score on the test-set.

---

## 🛠️ Requirements & Installation

Make sure you have Python 3.8+ installed. You can install all the required dependencies for both the MFWD tools and the CBIR engine using:

```bash
pip install numpy opencv-python nmslib scikit-image scikit-learn tensorflow keras matplotlib tqdm pandas albumentations timm pytorch
```

*(Note: The MFWD scripts also require standard libraries like `ftplib`, `pathlib`, and `argparse` which are included in Python by default).*

---

## 👥 Credits & Citation

**Author:**
- Sergio Beamonte González

**Original MFWD Dataset Citation:**
If you use the data downloading scripts or the MFWD dataset, please cite the original authors:

> Genze, N., Vahl, W.K., Groth, J. et al. Manually annotated and curated Dataset of diverse Weed Species in Maize and Sorghum for Computer Vision. Sci Data 11, 109 (2024). [https://doi.org/10.1038/s41597-024-02945-6](https://doi.org/10.1038/s41597-024-02945-6)

```bibtex
@article{Genze2024,
   abstract = {Sustainable weed management strategies are critical to feeding the world’s population while preserving ecosystems and biodiversity...},
   author = {Nikita Genze and Wouter K Vahl and Jennifer Groth and Maximilian Wirth and Michael Grieb and Dominik G Grimm},
   doi = {10.1038/s41597-024-02945-6},
   issn = {2052-4463},
   issue = {1},
   journal = {Scientific Data},
   pages = {109},
   title = {Manually annotated and curated Dataset of diverse Weed Species in Maize and Sorghum for Computer Vision},
   year = {2024}
}
```
