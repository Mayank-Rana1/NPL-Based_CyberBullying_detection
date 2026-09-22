# Cyberbullying Detection using NLP

A simple machine learning project to detect cyberbullying in text comments. The model classifies comments as **Bullying** or **Not Bullying** using natural language processing.

This project was made as part of our college minor project.

## Problem Statement

Social media platforms have seen a rise in hate speech and bullying. Manual moderation is not scalable. This project aims to automatically detect bullying comments using NLP and Machine Learning.

## Dataset

We used a cleaned and balanced dataset of **4,348 comments**:
- 2,174 bullying comments
- 2,174 non-bullying comments

The dataset was created by cleaning and balancing a larger 14k raw dataset, removing duplicates and adding real-world examples to handle cases like negation (e.g., `you will get success` vs `you will not get success`).

Columns:
- `text` - the comment
- `label` - 1 for bullying, 0 for not bullying

Dataset file: `dataset.csv`

## Tech Stack

- **Language:** Python 3.10+
- **Libraries:** pandas, scikit-learn
- **Model:** Logistic Regression with TF-IDF Vectorizer
- **Vectorizer:** TF-IDF with n-gram range (1,3) to capture phrases like "will not get"

We chose Logistic Regression because it works well for text classification and gives probability scores.

## Project Structure

```
CyberBullying/
│
├── dataset.csv       # Final cleaned and balanced dataset (4348 rows)
├── train.py          # Training script
├── app.py            # CLI app to test comments
├── model.pkl         # Trained model (generated after training)
├── vectorizer.pkl    # TF-IDF vectorizer (generated after training)
├── requirements.txt  # Dependencies
└── README.md         # Project documentation
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will:
- Load and clean the dataset
- Split data into train/test (80/20)
- Train TF-IDF + Logistic Regression
- Show accuracy, classification report and confusion matrix
- Save `model.pkl` and `vectorizer.pkl`

Expected output: Accuracy around 98-99% on test set.

### 3. Test the model

```bash
python app.py
```

Then type any comment:

```
Enter comment: you are a nice person
-> NOT BULLYING (96.6% confident)

Enter comment: you are a piece of trash
-> BULLYING DETECTED (85.9% confident)

Enter comment: you will get success
-> NOT BULLYING (99.2% confident)

Enter comment: you will not get success
-> BULLYING DETECTED (89.6% confident)
```

Type `exit` to quit.

## Results

On our final balanced dataset (4348 rows):

- **Accuracy:** ~99%
- **Bullying Precision:** 0.98, Recall: 1.00
- **Not Bullying Precision:** 1.00, Recall: 0.97

The model correctly handles:
- Typos like `peace of trash` (for `piece of trash`)
- Negation like `will get success` vs `will not get success`
- Common bullying phrases like `you are a burden`, `you will fail`

## Limitations

- The model works on English text only
- Very short or gibberish text may give low confidence
- Sarcasm and context-based bullying is hard to detect with TF-IDF

## Future Improvements

- Add support for Hindi/Hinglish comments
- Try BERT or other transformer models for better context understanding
- Create a web app using Flask/Streamlit
- Add real-time comment filtering for social media

## Author

**Mayank Rana**
B.Tech - Minor Project 2026

## License

This project is for educational purposes.

---
If you find this useful, give it a star ⭐
