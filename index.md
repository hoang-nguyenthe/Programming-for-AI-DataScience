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

In this assignment, our team performed comprehensive Exploratory Data Analysis across three distinct data modalities: **Tabular**, **Text**, and **Image**. The goal was to understand each dataset's structure, uncover hidden patterns, detect potential issues, and prepare insights for downstream AI/DS tasks.

### 📊 Datasets Used

| Modality | Dataset | Source | Description |
|:--------:|---------|:------:|-------------|
| 🗃️ Tabular | Advertisement — Click on Ad | [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad) | 1,000 user records predicting whether a user clicks on an online advertisement |
| 📝 Text | Stanford Question Answering Dataset (SQuAD 1.1) | [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad) | 100,000+ question-answer pairs based on Wikipedia articles |
| 🖼️ Image | Street View House Numbers (SVHN) | [Stanford](http://ufldl.stanford.edu/housenumbers/) | 99,289 real-world digit images (0–9) captured from Google Street View |

### 📓 Notebooks

| Modality | Notebook | Key Visualizations |
|:--------:|----------|-------------------|
| 🗃️ Tabular | [eda_tabular.ipynb](eda_tabular.ipynb) | Histograms, Boxplots, Correlation Heatmap, Pairplot, Time Analysis |
| 📝 Text | [eda_text.ipynb](eda_text.ipynb) | Length Distributions, Question Type Analysis, Word Cloud, Top Topics |
| 🖼️ Image | [eda_image.ipynb](eda_image.ipynb) | Sample Grid, Label Distribution, Mean Images, Pixel Intensity, Brightness |

### 🔍 Key Findings

**Tabular (Ad Click):**
- Perfectly balanced dataset (500 clicks vs 500 non-clicks).
- Users who click ads tend to be older, have lower income, and spend less time on the internet.
- Strong negative correlation between Daily Internet Usage and ad clicks (−0.79).

**Text (SQuAD):**
- 87,000+ QA pairs from 442 Wikipedia articles covering diverse topics.
- "What" questions dominate at 43.1%, followed by "How" and "Who".
- Answers are very short (mean ~3 words), suitable for extractive QA tasks.

**Image (SVHN):**
- Significant class imbalance — digit "1" has ~3x more samples than digit "9".
- Images are noisy with varying lighting conditions due to real-world street photography.
- All images are uniformly sized at 32×32 RGB pixels.

### 🔗 Links

| Item | Link |
|:----:|------|
| 📄 Report PDF | *Coming soon* |
| 🎥 Presentation Video | *Coming soon* |
| 💻 GitHub Repository | [View on GitHub](.) |

---

### 🛠️ Tools & Technologies

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `NLTK` · `WordCloud` · `PyTorch` · `HuggingFace Datasets` · `SciPy` · `Jupyter Notebook`

---

*© 2026 — P4AI-DS (CO3135), Department of Computer Science, HCMUT, VNU-HCM*
