# 🚀 P4AI-DS Assignment 1 — Exploratory Data Analysis (EDA)

[![Course](https://img.shields.io/badge/Course-CO3135-blue)]()
[![University](https://img.shields.io/badge/University-HCMUT-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Python](https://img.shields.io/badge/Python-3.13-yellow)]()
[![Modalities](https://img.shields.io/badge/Modalities-4%20(Tabular%20%7C%20Text%20%7C%20Image%20%7C%20Multimodal)-purple)]()

> **Programming for Artificial Intelligence and Data Science (P4AI-DS)**  
> Ho Chi Minh City University of Technology (HCMUT), VNU-HCM  
> Instructor: Dr. Thanh-Sach LE  
> Semester II — Academic Year 2025–2026

---

## 📌 Overview

This repository contains our team's work on **Assignment 1: Exploratory Data Analysis (EDA)** for the P4AI-DS course. We performed comprehensive EDA across **four data modalities** — **Tabular**, **Text**, **Image**, and **Multimodal (Text + Image)** — to understand, visualize, and draw actionable insights from real-world datasets.

🌐 **Live Landing Page:** [View on GitHub Pages](https://hoang-nguyenthe.github.io/Programming-for-AI-DataScience/)

---

## 👥 Team Members

| Student ID | Full Name | Role |
|:----------:|-----------|:----:|
| 2352354 | Nguyễn Thế Hoàng | 🔰 Team Leader |
| 2352179 | Ngô Trần Đình Duy | Member |
| 2352378 | Hồ Minh Huy | Member |
| 2352780 | Nguyễn Việt Nam | Member |

---

## 📂 Repository Structure

```
Programming-for-AI-DataScience/
│
├── index.md                                    # GitHub Pages landing page
├── README.md                                   # This file
│
├── eda_tabular.ipynb                           # 🗃️ Tabular EDA notebook
├── eda_text.ipynb                              # 📝 Text EDA notebook
├── eda_image.ipynb                             # 🖼️ Image EDA notebook
├── eda_multimodal.ipynb                        # 🎬 Multimodal EDA notebook
│
├── advertising.csv                             # Tabular dataset (Ad Click)
│
├── tabular_01_target_distribution.png          # Tabular visualizations
├── tabular_02_numerical_distributions.png
├── tabular_03_boxplot_by_target.png
├── tabular_04_correlation_heatmap.png
├── tabular_05_pairplot.png
├── tabular_06_time_analysis.png
├── tabular_07_country_gender.png
│
├── text_01_length_distributions.png            # Text visualizations
├── text_02_question_types.png
├── text_03_top_titles.png
├── text_04_wordcloud_questions.png
├── text_05_top_words_and_position.png
│
├── image_01_random_samples.png                 # Image visualizations
├── image_02_sample_grid.png
├── image_03_label_distribution.png
├── image_04_mean_images.png
├── image_05_pixel_distribution.png
├── image_06_brightness_and_comparison.png
├── image_07_rgb_gray_dark.png
│
├── multimodal_01_sample_images_captions.png    # Multimodal visualizations
├── multimodal_02_caption_lengths.png
├── multimodal_03_wordcloud_captions.png
├── multimodal_04_content_analysis.png
├── multimodal_05_image_dimensions.png
├── multimodal_06_color_analysis.png
├── multimodal_07_caption_agreement.png
└── multimodal_08_first_words.png
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
- **Size:** 87,599 training + 10,570 validation = 98,169 question-answer pairs
- **Task:** Reading comprehension — extractive question answering
- **Structure:** Each sample contains a Wikipedia context paragraph, a question, and an answer span

### 3. 🖼️ Image — Street View House Numbers (SVHN)
- **Source:** [Stanford](http://ufldl.stanford.edu/housenumbers/)
- **Size:** 73,257 training + 26,032 testing = 99,289 images
- **Task:** Digit classification (0–9)
- **Format:** 32×32 pixel RGB images captured from Google Street View

### 4. 🎬 Multimodal — Flickr8k (Text + Image)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Size:** 8,091 images × 5 captions = 40,455 image-caption pairs
- **Task:** Image captioning — describing images with natural language
- **Structure:** Each image has 5 independently written English captions

---

## 🔍 Key Findings

### 🗃️ Tabular (Ad Click)
- ✅ Perfectly balanced: 500 clicks vs 500 non-clicks
- 📉 Strong negative correlation: Daily Internet Usage ↔ Clicked on Ad (−0.79)
- 👤 Users who click ads: older, lower income, less internet usage
- 🧹 No missing values or duplicates
- 🎯 Clear class separation in pairplots → simple classifiers could perform well

### 📝 Text (SQuAD)
- ❓ "What" questions dominate at 43.1% of all questions
- 📏 Context averages ~120 words; answers are very short (~3 words)
- 🌍 Diverse topics: New York City, Super Bowl, Beyoncé, Buddhism, etc.
- 📍 Answers tend to appear early in the context paragraph
- 📚 98,169 QA pairs from 442 unique Wikipedia articles

### 🖼️ Image (SVHN)
- ⚖️ Class imbalance: digit "1" has ~3× more samples than digit "9"
- 🌤️ Wide brightness variation due to outdoor street photography
- 📐 Uniform size (32×32) — no resizing needed
- 🔢 Some images contain distracting digits at the edges
- 🎨 Pixel intensity: R=111.1, G=112.7, B=120.0 (blue slightly higher)

### 🎬 Multimodal (Flickr8k)
- 📸 40,455 image-caption pairs from 8,091 unique images
- 📝 Captions average 11.8 words; 61.1% start with "A" (e.g., "A dog runs...")
- 🐕 Most described subjects: dogs, men, women, children in outdoor scenes
- 🤝 Inter-annotator Jaccard similarity ≈ 0.278 (moderate agreement)
- 📐 Non-uniform image sizes (mainly 500×333 or 333×500 pixels)
- 🎨 Warm-toned images: R=117.3, G=114.4, B=103.3; mean brightness=111.7

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| NLP | NLTK, HuggingFace Datasets |
| Computer Vision | PyTorch, Torchvision, SciPy, Pillow |
| Environment | Jupyter Notebook, VS Code |
| Report | LaTeX (Overleaf) |
| Version Control | Git, GitHub, GitHub Pages |

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hoang-nguyenthe/Programming-for-AI-DataScience.git
   cd Programming-for-AI-DataScience
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud nltk datasets torch torchvision scipy pillow kaggle
   ```

3. **Run the notebooks:**
   Open each `.ipynb` file in Jupyter Notebook or VS Code and click **Run All**.

   | Notebook | Dataset | Download |
   |----------|---------|----------|
   | `eda_tabular.ipynb` | Ad Click | ✅ Included (`advertising.csv`) |
   | `eda_text.ipynb` | SQuAD | 🔄 Auto-downloads from HuggingFace |
   | `eda_image.ipynb` | SVHN | 🔄 Auto-downloads from Stanford |
   | `eda_multimodal.ipynb` | Flickr8k | 🔄 Auto-downloads via Kaggle API* |

   *\*Requires [Kaggle API token](https://www.kaggle.com/settings) — see notebook for setup instructions.*

---

## 📎 Links

| Item | Link |
|:----:|------|
| 🌐 Landing Page | [GitHub Pages](https://hoang-nguyenthe.github.io/Programming-for-AI-DataScience/) |
| 📄 Report PDF | *Coming soon* |
| 🎥 Video Presentation | *Coming soon* |

---

## 📜 References

1. Gabriel Santello. *Advertisement - Click on Ad Dataset.* Kaggle, 2021. [Link](https://www.kaggle.com/datasets/gabrielsantello/advertisement-click-on-ad)
2. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. *SQuAD: 100,000+ Questions for Machine Comprehension of Text.* EMNLP, 2016. [Link](https://rajpurkar.github.io/SQuAD-explorer/)
3. Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. *Reading Digits in Natural Images with Unsupervised Feature Learning.* NIPS Workshop, 2011. [Link](http://ufldl.stanford.edu/housenumbers/)
4. Hodosh, M., Young, P., & Hockenmaier, J. *Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics.* JAIR, 2013. [Link](https://www.kaggle.com/datasets/adityajn105/flickr8k)
5. McKinney, W. *Data Structures for Statistical Computing in Python (Pandas).* SciPy, 2010.
6. Hunter, J. D. *Matplotlib: A 2D Graphics Environment.* CSE, 2007.
7. Waskom, M. L. *seaborn: statistical data visualization.* JOSS, 2021.
8. Paszke, A., et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS, 2019.

---

## ⚖️ Academic Integrity

This work was completed in accordance with HCMUT's academic integrity policy. All datasets and external resources are properly cited. Code was developed by the team members with the assistance of publicly available documentation and tutorials.

---

*© 2026 — P4AI-DS (CO3135), Department of Computer Science, HCMUT, VNU-HCM*
