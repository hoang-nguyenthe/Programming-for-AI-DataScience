# 🚀 P4AI-DS Assignment 1 — Exploratory Data Analysis (EDA)

[![Course](https://img.shields.io/badge/Course-CO3135-blue)]()
[![University](https://img.shields.io/badge/University-HCMUT-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.13-yellow)]()

> **Programming for Artificial Intelligence and Data Science (P4AI-DS)**
> Ho Chi Minh City University of Technology (HCMUT), VNU-HCM
> Instructor: Dr. Thanh-Sach LE

---

## 📌 Overview

This repository contains our team's work on **Assignment 1: Exploratory Data Analysis (EDA)** for the P4AI-DS course. We performed comprehensive EDA across three data modalities — **Tabular**, **Text**, and **Image** — to understand, visualize, and draw actionable insights from real-world datasets.

🌐 **Live Landing Page:** [View on GitHub Pages](https://github.com/hoang-nguyenthe/Programming-for-AI-DataScience)

---

## 👥 Team Members

| Student ID | Full Name | Role |
|:----------:|-----------|:----:|
| 2352354 | Nguyễn Thế Hoàng | Team Leader |
| 2352179 | Ngô Trần Đình Duy | Member |
| 2352378 | Hồ Minh Huy | Member |
| 2352780 | Nguyễn Việt Nam | Member |

---

## 📂 Repository Structure

```
p4ai-eda/
│
├── index.md                          # GitHub Pages landing page
├── README.md                         # This file
│
├── eda_tabular.ipynb                 # 🗃️ Tabular EDA notebook
├── eda_text.ipynb                    # 📝 Text EDA notebook
├── eda_image.ipynb                   # 🖼️ Image EDA notebook
│
├── advertising.csv                   # Tabular dataset (Ad Click)
│
├── tabular_01_target_distribution.png
├── tabular_02_numerical_distributions.png
├── tabular_03_boxplot_by_target.png
├── tabular_04_correlation_heatmap.png
├── tabular_05_pairplot.png
├── tabular_06_time_analysis.png
├── tabular_07_country_gender.png
│
├── text_01_length_distributions.png
├── text_02_question_types.png
├── text_03_top_titles.png
├── text_04_wordcloud_questions.png
├── text_05_top_words_and_position.png
│
├── image_01_random_samples.png
├── image_02_sample_grid.png
├── image_03_label_distribution.png
├── image_04_mean_images.png
├── image_05_pixel_distribution.png
├── image_06_brightness_and_comparison.png
└── image_07_rgb_gray_dark.png
```

---

## 📊 Datasets

### 1. 🗃️ Tabular — Advertisement Click on Ad
- **Source:** [Kaggle](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad)
- **Size:** 1,000 records × 10 features
- **Task:** Predict whether a user will click on an online advertisement
- **Key Features:** Daily Time Spent on Site, Age, Area Income, Daily Internet Usage, Gender

### 2. 📝 Text — Stanford Question Answering Dataset (SQuAD 1.1)
- **Source:** [HuggingFace](https://huggingface.co/datasets/rajpurkar/squad)
- **Size:** 87,599 training + 10,570 validation question-answer pairs
- **Task:** Reading comprehension — extractive question answering
- **Structure:** Each sample contains a Wikipedia context paragraph, a question, and an answer span

### 3. 🖼️ Image — Street View House Numbers (SVHN)
- **Source:** [Stanford](http://ufldl.stanford.edu/housenumbers/)
- **Size:** 73,257 training + 26,032 testing images
- **Task:** Digit classification (0–9)
- **Format:** 32×32 pixel RGB images captured from Google Street View

---

## 🔍 Key Findings

### Tabular (Ad Click)
- ✅ Perfectly balanced: 500 clicks vs 500 non-clicks
- 📉 Strong negative correlation: Daily Internet Usage ↔ Clicked on Ad (−0.79)
- 👤 Users who click ads: older, lower income, less internet usage
- 🧹 No missing values or duplicates

### Text (SQuAD)
- ❓ "What" questions dominate at 43.1% of all questions
- 📏 Context averages ~120 words; answers are very short (~3 words)
- 🌍 Diverse topics: New York City, Super Bowl, Beyoncé, Buddhism, etc.
- 📍 Answers tend to appear early in the context paragraph

### Image (SVHN)
- ⚖️ Class imbalance: digit "1" has ~3× more samples than digit "9"
- 🌤️ Wide brightness variation due to outdoor street photography
- 📐 Uniform size (32×32) — no resizing needed
- 🔢 Some images contain distracting digits at the edges

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| NLP | NLTK, HuggingFace Datasets |
| Computer Vision | PyTorch, Torchvision, SciPy |
| Environment | Jupyter Notebook, VS Code |

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/p4ai-eda.git
   cd p4ai-eda
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud nltk datasets torch torchvision scipy
   ```

3. **Run the notebooks:**
   Open each `.ipynb` file in Jupyter Notebook or VS Code and click **Run All**.

   - `eda_tabular.ipynb` — requires `advertising.csv` in the same directory
   - `eda_text.ipynb` — automatically downloads SQuAD from HuggingFace
   - `eda_image.ipynb` — automatically downloads SVHN from Stanford

---

## 📎 Links

| Item | Link |
|:----:|------|
| 🌐 Landing Page | [GitHub Pages](https://github.com/hoang-nguyenthe/Programming-for-AI-DataScience) |
| 📄 Report PDF | *Coming soon* |
| 🎥 Video Presentation | *Coming soon* |

---

## 📜 References

- Gabriel Santello. *Advertisement - Click on Ad Dataset.* Kaggle, 2021.
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP, 2016.
- Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. *Reading Digits in Natural Images with Unsupervised Feature Learning.* NIPS Workshop, 2011.

---

## ⚖️ Academic Integrity

This work was completed in accordance with HCMUT's academic integrity policy. All datasets and external resources are properly cited. Code was developed by the team members with the assistance of publicly available documentation and tutorials.

---

*© 2026 — P4AI-DS (CO3135), Department of Computer Science, HCMUT, VNU-HCM*
