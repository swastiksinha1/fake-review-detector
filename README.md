<div align="center">

# 🕵️‍♂️ Fake Review Detector  
### Unmasking E-Commerce Spam

### A Hybrid ML Approach Against AI-Generated Reviews

An Applied Machine Learning project trained on 40,000 Amazon reviews, designed to detect deceptive reviews by analyzing not just what is written, but how it is written.

</div>

---

## 📌 Overview

In today’s AI-driven internet, generating highly convincing fake reviews has become easier than ever. Large Language Models can now create realistic product feedback in seconds, making traditional spam detection systems increasingly ineffective. 

The **Fake Review Detector** is a hybrid ML application that looks beyond basic word frequencies. It combines traditional NLP with a custom behavioral feature engine to spot the "stylistic fingerprints" left behind by generative AI and paid human reviewers, drastically reducing false positives on genuinely enthusiastic feedback.

---

## 🚀 Features

| Feature | Description |
| :--- | :--- |
| 🧠 **Hybrid Prediction Engine** | Combines TF-IDF, Behavioral Extraction, and Rule-Based Heuristics |
| 📐 **29 Stylistic Features** | Analyzes caps ratios, punctuation abuse, and sentiment inconsistency |
| 🔍 **AI Fingerprint Detection** | Flags excessive positivity, persuasive language, and lack of product specifics |
| ⚡ **Real-Time Analysis** | Instant fake vs. real probability scoring via web interface |
| 📱 **Premium UI/UX** | Custom 3D Glassmorphism interface with animated fluid backgrounds |
| 📦 **Batch Processing** | Analyze multiple reviews simultaneously |

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Machine Learning** | scikit-learn, pandas, numpy, scipy |
| **Backend & Deployment** | Python, Streamlit Community Cloud |
| **Frontend / UI** | Streamlit, Custom CSS |
| **Data Processing** | Regex, Custom Feature Engineering |

---

## 🎯 Problem Statement

Traditional spam detectors face major difficulties while:

- Catching modern AI-generated reviews that use natural, realistic vocabulary.
- Avoiding false positives on real, highly enthusiastic 5-star reviews.
- Identifying long-winded, verbose praise that lacks concrete product specifications.
- Spotting sentiment contradictions (e.g., 1-star ratings with glowing text).

The Fake Review Detector solves these issues by shifting the focus from *content analysis* to *behavioral and stylistic analysis*.

---

## 📊 Model Performance

| Metric | Score |
| :--- | :--- |
| **Best Model** | Logistic Regression (C=2) |
| **Accuracy** | 91.6% |
| **F1-Score** | 91.6% |
| **AUC-ROC** | 97.6% |
| **Cross-Validation F1** | 0.9104 ± 0.0071 |

---

## 📂 Project Structure

```bash
fake-review-detector
│
├── 📁 data/               # Dataset (40,000 Amazon reviews)
├── 📁 model/              # Trained .pkl files (model, tfidf, scaler, meta)
│
├── 📄 app.py              # Main Streamlit web application
├── 📄 train.py            # ML model training pipeline
├── 📄 utils.py            # Feature engineering & heuristics logic
├── 📄 analyze_data.py     # Exploratory Data Analysis scripts
├── 📄 test_model.py       # Validation and testing scripts
│
├── 📄 requirements.txt    # Project dependencies
└── 📄 README.md           # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository

Bash
git clone [https://github.com/swastiksinha1/fake-review-detector.git](https://github.com/swastiksinha1/fake-review-detector.git)

2️⃣ Navigate to the Project Folder
Bash
cd fake-review-detector

3️⃣ Install Dependencies
Bash
pip install -r requirements.txt

4️⃣ Run the Web App
Bash
streamlit run app.py

(Note: If you haven't trained the model yet, run python train.py first to generate the necessary .pkl files in the model/ directory).


👨‍💻 Developer
Swastik Sinha
Full Stack Developer • VIT Bhopal

📈 Learning Outcomes
This project helped in gaining practical experience in:

Applied Machine Learning & NLP
Advanced Feature Engineering
Full-Stack UI/UX Design with Streamlit & CSS
Model Evaluation (AUC-ROC, Cross-Validation)
Cloud Deployment & Dependency Management
Problem Solving using Technology

⭐ Support
If you liked this project:

⭐ Star the repository
🍴 Fork the project
🛠️ Contribute to improvements
🧪 Test edge-cases on the live demo!

📄 License
This project is licensed under the MIT License.

🏷️ GitHub Topics

machine-learning
nlp
python
streamlit
data-science
artificial-intelligence
cybersecurity
feature-engineering
vit-bhopal
open-source



