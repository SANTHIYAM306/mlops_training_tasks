import joblib

print("========================================")
print("🧠 OMNIGUARD: BRAIN X-RAY 🧠")
print("========================================")

# 1. Load the Brain
vectorizer = joblib.load('omniguard_vectorizer.joblib')
model = joblib.load('omniguard_model.joblib')

# 2. The exact message that tricked the AI
sneaky_scam = "Subject: Congratulations! You’ve Won ₹100000 Dear User,We are excited to inform you that you have won ₹100000 in our lucky draw contest!To claim your reward, please click the link below and complete the verification process:Click Here to ContinueHurry! This offer is valid for a limited time only.Congratulations once again!Best Regards,Reward Team"

# 3. Translate to math
math_input = vectorizer.transform([sneaky_scam])

# 4. Extract the exact math weights from the SVM
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# 5. Find only the words that actually appeared in the email
words_in_email = math_input.nonzero()[1]

print("Analyzing the exact mathematical weight of each word...")
print("-> POSITIVE numbers mean the AI thinks it's a SCAM.")
print("-> NEGATIVE numbers mean the AI thinks it's SAFE.\n")

total_score = 0
for index in words_in_email:
    word = feature_names[index]
    score = coefficients[index]
    total_score += score
    
    # We only want to print words that the AI actually recognized
    if score > 0:
        print(f"🚨 '{word}': {score:.4f} (Pushing towards SCAM)")
    elif score < 0:
        print(f"✅ '{word}': {score:.4f} (Pushing towards SAFE)")
print("-" * 40)
bias = model.intercept_[0]
true_final_math = total_score + bias

print(f"1. Sum of the Words (wx): {total_score:.4f}")
print(f"2. The Hidden Model Bias (b): {bias:.4f}")
print("-" * 40)
print(f"TRUE FINAL MATH (wx + b): {true_final_math:.4f}")

if true_final_math > 0:
    print("\n🚨 ACTUAL VERDICT: SCAM")
else:
    print("\n✅ ACTUAL VERDICT: SAFE")