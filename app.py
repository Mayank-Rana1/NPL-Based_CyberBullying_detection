import pickle
import os

# Load model
if not os.path.exists('model.pkl'):
    print("Model not found! Run python train.py first")
    exit()

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

print("=== NLP-Based Cyberbullying Detection ===")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter comment: ")
    if text.lower() == 'exit':
        break
    if not text.strip():
        continue

    vec = vectorizer.transform([text.lower()])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    if pred == 1:
        print(f"  -> ⚠️  BULLYING DETECTED ({prob*100:.1f}% confident)")
    else:
        print(f"  -> ✅ NOT BULLYING ({prob*100:.1f}% confident)")
