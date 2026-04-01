# 🎓 P4AI-DS (CO3135) — Assignment Landing Page

> **Programming for Artificial Intelligence and Data Science**  
> Ho Chi Minh City University of Technology (HCMUT), VNU-HCM  
> Semester II — Academic Year 2025–2026

---

## 👥 Team Members

| Student ID | Full Name | Role |
|:----------:|-----------|:----:|
| 2352354 | Nguyễn Thế Hoàng | Team Leader |
| 2352179 | Ngô Trần Đình Duy | Member |
| 2352378 | Hồ Minh Huy | Member |
| 2352780 | Nguyễn Việt Nam | Member |

---

## 📋 Assignment 1 — Exploratory Data Analysis (EDA)

In this assignment, our team performed comprehensive Exploratory Data Analysis across **four distinct data modalities**: Tabular, Text, Image, and Multimodal (Text + Image). The goal was to understand each dataset's structure, uncover hidden patterns, detect potential issues, and prepare insights for downstream AI/DS tasks.

### 📊 Datasets Used

| Modality | Dataset | Source | Description |
|:--------:|---------|:------:|-------------|
| 🗃️ Tabular | Advertisement — Click on Ad | [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad) | 1,000 user records predicting ad clicks |
| 📝 Text | SQuAD 1.1 | [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad) | 98,169 question-answer pairs from Wikipedia |
| 🖼️ Image | SVHN | [Stanford](http://ufldl.stanford.edu/housenumbers/) | 99,289 real-world digit images (0–9) from Google Street View |
| 🎬 Multimodal | Flickr8k | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) | 8,091 images × 5 captions = 40,455 image-caption pairs |

### 📓 Notebooks

| Modality | Notebook | Key Visualizations |
|:--------:|----------|-------------------|
| 🗃️ Tabular | [eda_tabular.ipynb](eda_tabular.ipynb) | Histograms, Boxplots, Correlation Heatmap, Pairplot, Time Analysis |
| 📝 Text | [eda_text.ipynb](eda_text.ipynb) | Length Distributions, Question Type Analysis, Word Cloud, Top Topics |
| 🖼️ Image | [eda_image.ipynb](eda_image.ipynb) | Sample Grid, Label Distribution, Mean Images, Pixel Intensity |
| 🎬 Multimodal | [eda_multimodal.ipynb](eda_multimodal.ipynb) | Sample Images+Captions, Word Cloud, Caption Agreement, Image Dimensions |

### 🔍 Key Findings

**🗃️ Tabular (Ad Click):**
- Perfectly balanced dataset (500 clicks vs 500 non-clicks).
- Users who click ads tend to be older, have lower income, and spend less time on the internet.
- Strong negative correlation between Daily Internet Usage and ad clicks (−0.79).
- Clear class separation suggests simple classifiers could perform well.

**📝 Text (SQuAD):**
- 98,169 QA pairs from 442 Wikipedia articles covering diverse topics.
- "What" questions dominate at 43.1%, followed by "How" and "Who" (9.4% each).
- Answers are very short (mean ~3 words), suitable for extractive QA tasks.
- Answers tend to appear early in the context paragraph.

**🖼️ Image (SVHN):**
- Significant class imbalance — digit "1" has ~3× more samples than digit "9".
- Images are noisy with varying lighting conditions due to real-world street photography.
- All images uniformly sized at 32×32 RGB pixels.
- Pixel intensity: R=111.1, G=112.7, B=120.0.

**🎬 Multimodal (Flickr8k):**
- 40,455 image-caption pairs from 8,091 unique images.
- Captions average 11.8 words; 61.1% start with "A" (e.g., "A dog runs...").
- Most described subjects: dogs, men, women, children in outdoor scenes.
- Inter-annotator Jaccard similarity ≈ 0.278 (moderate word overlap between captions).
- Non-uniform image sizes (mainly 500×333 or 333×500); warm-toned images (R=117.3, G=114.4, B=103.3).

### 🔗 Links

| Item | Link |
|:----:|------|
| 📄 Report PDF | *Coming soon* |
| 🎥 Presentation Video | *Coming soon* |
| 💻 GitHub Repository | [View on GitHub](https://github.com/hoang-nguyenthe/Programming-for-AI-DataScience) |

---

### 🛠️ Tools & Technologies

`Python 3.13` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `NLTK` · `WordCloud` · `PyTorch` · `Torchvision` · `HuggingFace Datasets` · `SciPy` · `Pillow` · `Kaggle API` · `LaTeX` · `Overleaf`

---

*© 2026 — P4AI-DS (CO3135), Department of Computer Science, HCMUT, VNU-HCM*
