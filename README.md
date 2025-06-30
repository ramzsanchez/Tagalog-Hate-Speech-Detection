# Hate Speech Detection in Tagalog using XGBoost, RoBERTa, and Ensemble Learning

This repository contains the source code, datasets, and models related to our undergraduate thesis:  
**"Leveraging XGBoost, RoBERTa, and Ensemble Learning for Hate Speech Detection in Tagalog"**  
by **Ramil Regen Sanchez, Jemuel Gen Medroso, and Prince Lloyd Cagampang**  
submitted to Caraga State University, June 2023.

## 📌 Abstract

This project addresses the growing challenge of detecting hate speech in **Tagalog**, a low-resource language, using machine learning and natural language processing techniques. We leverage:

- [XGBoost](https://xgboost.readthedocs.io/en/stable/)
- [RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)
- An **ensemble model** combining both with [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) as a meta-learner.

Our ensemble model outperforms individual models in terms of accuracy, precision, recall, and F1-score when tested on a hate speech dataset derived from Twitter data.

## 🔍 Features

- Preprocessing using [Pinoy Tweetokenize](https://github.com/jcblaisecruz02/pinoy-tweetokenize)
- Feature extraction via [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- Transfer learning using pretrained [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) models via the [Transformers library](https://huggingface.co/transformers/)
- Ensemble Learning with [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- Statistical evaluation using [McNemar’s Test](https://en.wikipedia.org/wiki/McNemar%27s_test) and [Cohen's d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)


## 📊 Results

| Model             | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| XGBoost-TagHS     | 74.41%   | 75.27%    | 68.56% | 71.76%   |
| RoBERTa-TagHS     | 77.65%   | 77.36%    | 74.74% | 76.03%   |
| XRoBERTa-SLR (Ensemble) | **87.64%** | **87.44%** | **86.35%** | **86.89%** |

[XGBoost-TagHS Model](https://huggingface.co/ramgensanchez/XGBoost-Tagalog-Hate-Speech)
[RoBERTa-TagHS Model](https://huggingface.co/ramgensanchez/RoBERTa-Tagalog-Hate-Speech)
[XRoBERTa-SLR (Ensemble) Model](https://huggingface.co/ramgensanchez/XRoBERTa-SLR-Tagalog-Hate-Speech)

## 🛠️ Technologies Used

- Python 3.11
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [Transformers (by Hugging Face)](https://huggingface.co/transformers/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)

## 🧪 Training Environment

- CPU: AMD Ryzen 5800x
- RAM: 32GB
- GPU: NVIDIA RTX 3060Ti (8GB)
- OS: Windows

## 🔗 Citation and Dataset Source

- [Filipino Hate Speech Dataset](https://huggingface.co/datasets/jcblaise/filipino-hate-speech) by Cabasag et al., 2019
- [RoBERTa Pretrained Model](https://huggingface.co/models)

## 🤝 Acknowledgments

Special thanks to:

- **Clark Justine Gonzales** (Thesis Adviser)
- **Dr. Jaymer Jayoma, Mark Phil Pacot, Giovanni Esma** (Panel Members)
- Our families, friends, and mentors for their unwavering support.

---

📬 For any inquiries, reach out via email or submit an issue in this repository.

