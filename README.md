# NLP-Based Cyberbullying Detection - Simple MVP

**No API | 100% Offline | Human-Readable Code | <100 Lines**

## What is this?
Detects if a comment is bullying or not using NLP.

## Folder Structure
- dataset.csv - 40 sample comments (1=Bullying, 0=Not Bullying)
- train.py - Trains TF-IDF + Naive Bayes model
- app.py - CLI version for testing
- app_gui.py - Tkinter GUI version (MVP Demo)
- requirements.txt

## How to Run (2 mins)

1. Install:
   pip install -r requirements.txt

2. Train:
   python train.py
   -> Creates model.pkl, vectorizer.pkl
   -> Shows Accuracy ~85-95%

3. Test:
   python app.py
   OR
   python app_gui.py

## Example
Input: "you are so dumb"
Output: BULLYING DETECTED

Input: "you are awesome"
Output: NOT BULLYING

## For PPT
Architecture: Input -> Lowercase -> TF-IDF -> Naive Bayes -> Output
Tech: Python, scikit-learn, Tkinter

## Future Scope
- Use BERT
- Add Hindi/ Hinglish support
- Real-time Instagram comment filter
