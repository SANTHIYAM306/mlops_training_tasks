import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import joblib

print("--- OMNIGUARD: TRAINING ON CUSTOM LOCAL DATA ---")

# 1. Load YOUR local dataset (Now letting Pandas read the header automatically!)
print("1. Loading 'combined_text_dataset.csv' from your folder...")
df = pd.read_csv('combined_text_dataset.csv')

# Drop any rows that are completely empty
df = df.dropna(subset=['label', 'text'])

# 2. Fix the Labels (Adding 'genuine' to the dictionary!)
label_mapping = {
    'genuine': 0, 'ham': 0, 'safe': 0, 'legit': 0, 
    'spam': 1, 'scam': 1, 'phishing': 1
}

# Make sure all labels are lowercase so they match perfectly, then convert to math
df['label'] = df['label'].astype(str).str.lower().str.strip().map(label_mapping)

# Drop any rows where the label still wasn't recognized
df = df.dropna(subset=['label'])

print(f"Successfully loaded {len(df)} rows of custom data!")

# 3. Train/Test Split
print("2. Splitting data into Training (80%) and Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 4. TF-IDF Vectorization
print("3. Converting text into mathematical matrices (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_math = vectorizer.fit_transform(X_train)
X_test_math = vectorizer.transform(X_test)

# 5. Train the Optimized SVM Model
print("4. Training the Lightning-Fast SVM (LinearSVC) on your data...")
svm_model = LinearSVC(random_state=42, dual=False)
svm_model.fit(X_train_math, y_train)

# 6. Evaluate the Final Exam
predictions = svm_model.predict(X_test_math)
accuracy = accuracy_score(y_test, predictions) * 100
print(f"\n🏆 Custom SVM Accuracy on {len(df)} rows: {accuracy:.2f}%")

# 7. Save the custom Brain
print("\n--- OMNIGUARD: FREEZING THE CUSTOM BRAIN ---")
joblib.dump(svm_model, 'omniguard_model.joblib')
joblib.dump(vectorizer, 'omniguard_vectorizer.joblib')
print("✅ Your custom AI Brain is saved and ready!")