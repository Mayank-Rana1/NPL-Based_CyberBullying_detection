import tkinter as tk
import pickle
import os

# Load model
model = None
vectorizer = None
if os.path.exists('model.pkl'):
    model = pickle.load(open('model.pkl','rb'))
    vectorizer = pickle.load(open('vectorizer.pkl','rb'))

def check_text():
    txt = entry.get("1.0", tk.END).strip()
    if not txt:
        return
    if model is None:
        result_label.config(text="Run train.py first!", fg="red")
        return
    vec = vectorizer.transform([txt.lower()])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    if pred == 1:
        result_label.config(text=f"⚠️ BULLYING DETECTED ({prob*100:.0f}%)", fg="white", bg="#e74c3c")
    else:
        result_label.config(text=f"✅ NOT BULLYING ({prob*100:.0f}%)", fg="white", bg="#2ecc71")

root = tk.Tk()
root.title("Cyberbullying Detection - MVP")
root.geometry("500x400")
root.config(bg="#f5f6fa")

tk.Label(root, text="NLP-Based Cyberbullying Detection", font=("Arial", 14, "bold"), bg="#f5f6fa").pack(pady=15)
entry = tk.Text(root, height=6, width=55, font=("Arial", 11))
entry.pack(pady=10)

tk.Button(root, text="Check", command=check_text, font=("Arial", 12, "bold"), bg="#0984e3", fg="white", width=20).pack(pady=10)

result_label = tk.Label(root, text="Result will appear here", font=("Arial", 13, "bold"), bg="#f5f6fa", fg="#2d3436", width=35, height=2)
result_label.pack(pady=20)

tk.Label(root, text="No API • Offline • TF-IDF + Naive Bayes", font=("Arial", 9), bg="#f5f6fa", fg="gray").pack(side=tk.BOTTOM, pady=10)

root.mainloop()
