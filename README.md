# KLCs - Keypoint Labeling Classifiers

This repository contains code and experiments for automatic keypoint description and identification for ViT-based models by using vision-language models, aiming to enhancing the explainability and interpretability in fine-grained image classification tasks. The approach is evaluated on the [CUB-200-2011](https://www.kaggle.com/datasets/wenewone/cub2002011) dataset.


## 📁 Dataset

Before running the code, download the **CUB-200-2011** dataset from Kaggle:

🔗 [https://www.kaggle.com/datasets/wenewone/cub2002011](https://www.kaggle.com/datasets/wenewone/cub2002011)

After downloading, extract the dataset and place it in a known location such as: 
```
./CUB_200_data/CUB_200_2011/
```
> 💡 Make sure to update the paths in the notebooks or scripts based on your local setup.

---

## ⚙️ Environment Setup

We recommend using a virtual environment for dependency management.

### 1. Clone the Repository

```bash
git clone https://github.com/chensy618/SuperpixelCUB.git
cd VLPart
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r clean_requirements_updated.txt
```

---

## 📈 Main Results

The primary experiments and visualizations can be found in the following Jupyter Notebooks:

- [`KCConCUB_update.ipynb`](KCConCUB_update.ipynb)  
  → Displays keypoint discovery, semantic alignment, and classification accuracy.

- [`SuperpixelInvestigationCUB_vlpart.ipynb`](SuperpixelInvestigationCUB_vlpart.ipynb)  
  → Explores the relationship between discovered prototypes and superpixel segments.

- Note: Since the notebook file is large, please download it first to view the results locally.
---

## ⚠️ Execution Notes

You may encounter errors such as:

```
FileNotFoundError: [Errno 2] No such file or directory
```

These are usually caused by incorrect file paths. Make sure all paths reflect your environment (e.g., local folder or mounted Google Drive).

---

## ✨ Highlights

- No training required on CUB for strong keypoint and semantic alignment.
- Achieves **82.7% classification accuracy with only 3 prototypes per class**.
- Supports **ViT-based** feature extractors.
-  Provides interpretable keypoint visualizations that help explain the model’s decision-making process.

---

