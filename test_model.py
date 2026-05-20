"""
Test script: Validates the trained model against known fake & real review patterns.
Run after train.py to verify the model catches tricky cases.
"""
import pickle
import numpy as np
import scipy.sparse as sp
from utils import clean_text, extract_behavioural_features, hybrid_predict

# Load model artifacts
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/meta.pkl", "rb") as f:
    meta = pickle.load(f)

print(f"Model: {meta['model_name']}")
print(f"Accuracy: {meta['accuracy']:.1%}  |  F1: {meta['f1']:.1%}  |  AUC: {meta.get('auc',0):.1%}")
print()


def predict(text, rating=3.0):
    pred, fake_prob, real_prob = hybrid_predict(text, rating, model, tfidf, scaler)
    return pred, fake_prob, real_prob


# ── Test cases ────────────────────────────────────────────────
# Each: (review_text, star_rating, expected_label)
# expected_label: "fake" or "real"
test_cases = [
    # === OBVIOUS FAKES (should be caught) ===
    (
        "This is the best product ever!!! Amazing!!! I love it so much!!! "
        "Everyone should buy this!!! Five stars!!!",
        5.0, "fake"
    ),
    (
        "Amazing product! Changed my life! Must buy! Absolutely incredible! "
        "Best thing I ever purchased! Love love love it!",
        5.0, "fake"
    ),
    (
        "Perfect perfect perfect. Best ever. Amazing. Love it. Buy now.",
        5.0, "fake"
    ),
    (
        "Great product great quality great price great everything great great great",
        5.0, "fake"
    ),
    (
        "I received this product for free in exchange for my honest review. "
        "It is absolutely amazing and wonderful! Five stars!",
        5.0, "fake"
    ),
    (
        "WOW THIS IS THE BEST PRODUCT I HAVE EVER BOUGHT IN MY ENTIRE LIFE "
        "YOU NEED TO BUY THIS RIGHT NOW",
        5.0, "fake"
    ),
    (
        "Awesome awesome awesome. Highly recommend to everyone. Must have.",
        5.0, "fake"
    ),
    (
        "good",
        5.0, "fake"
    ),
    (
        "Five stars. Perfect. Love it.",
        5.0, "fake"
    ),
    (
        "This product is amazing and I absolutely love it. "
        "I would definitely recommend it to all my friends and family. "
        "It has changed my life completely. Best purchase ever!",
        5.0, "fake"
    ),

    # === GENUINE-LOOKING REVIEWS (should be marked real) ===
    (
        "I bought this blender about 3 months ago. It works well for smoothies "
        "and soups. The motor is powerful enough for ice. My only complaint is "
        "the lid doesn't seal perfectly — sometimes liquid splashes if I fill "
        "it too much. Overall decent for the price.",
        4.0, "real"
    ),
    (
        "The book arrived on time and in good condition. The story was engaging "
        "though the ending felt a bit rushed. I'd recommend it to fans of "
        "the genre but it's not the best I've read this year.",
        3.0, "real"
    ),
    (
        "These headphones have decent sound quality for the price. Bass is a "
        "bit weak compared to my Sony pair but they're comfortable for long "
        "sessions. Battery lasts about 20 hours which matches their claim. "
        "The case feels a little cheap though.",
        4.0, "real"
    ),
    (
        "Returned this after a week. The stitching on the shoulder strap came "
        "apart after two uses. Material looks nice in photos but feels flimsy "
        "in person. Disappointed because the design was exactly what I wanted.",
        2.0, "real"
    ),
    (
        "Got this for my daughter's birthday. She loves it and plays with it "
        "every day. Assembly took about 30 minutes and the instructions were "
        "clear. Only giving 4 stars because one of the stickers was slightly "
        "misaligned out of the box.",
        4.0, "real"
    ),

    # === AI-GENERATED FAKES (ChatGPT-style) ===
    (
        "Unbelievable performance for the price. Don't even think twice.",
        5.0, "fake"
    ),
    (
        "Hands down the best purchase I've made this year. "
        "You won't regret it. Trust me.",
        5.0, "fake"
    ),
    (
        "Exceeded all my expectations. Phenomenal quality, flawless design. "
        "Look no further, this is the one.",
        5.0, "fake"
    ),

    # === MORE GENUINE REVIEWS ===
    (
        "Decent vacuum for a small apartment. Suction is good on hardwood "
        "but struggles a bit on thick carpet. The dustbin is 0.5L which "
        "means frequent emptying. Cord is about 15 feet — wish it were longer.",
        3.0, "real"
    ),

    # === SMART-QUOTE TEST (pasted from ChatGPT/Word) ===
    (
        "Unbelievable performance for the price. Don\u2019t even think twice.",
        5.0, "fake"
    ),

    # === VERBOSE AI MARKETING FAKE ===
    (
        "What an incredible purchase! I am beyond satisfied with this product "
        "and the entire buying experience. The materials feel premium, the "
        "performance is top-class, and the overall value for money is unbeatable. "
        "I’ve used many competing brands before, but this one clearly stands "
        "above the rest. The company has clearly invested a lot into making "
        "sure customers get the best possible experience. Even after heavy use, "
        "it still performs perfectly without any issues. I’m planning to buy "
        "another one soon as a gift because I’m that impressed.",
        5.0, "fake"
    ),

    # === VERBOSE AI MARKETING FAKE 2 (Generic entity + time bypass) ===
    (
        "I was a little hesitant before ordering, but after using this product "
        "for a few days, I can confidently say it was an amazing decision. The "
        "build quality, performance, and overall feel are simply outstanding. "
        "It performs smoothly without any issues and delivers exactly what was "
        "promised. I especially loved how easy it was to set up and start using "
        "immediately. This company clearly values customer satisfaction because "
        "every detail feels carefully designed. I’m extremely happy with my "
        "purchase and would definitely buy from this brand again",
        5.0, "fake"
    ),
]

# ── Run tests ─────────────────────────────────────────────────
print("=" * 70)
print("FAKE REVIEW DETECTION TEST SUITE")
print("=" * 70)

passed = 0
failed = 0
results = []

for i, (text, rating, expected) in enumerate(test_cases, 1):
    pred, fake_prob, real_prob = predict(text, rating)
    predicted = "fake" if pred == 1 else "real"
    correct = predicted == expected
    status = "PASS" if correct else "FAIL"

    if correct:
        passed += 1
    else:
        failed += 1

    preview = text[:60] + "..." if len(text) > 60 else text
    print(f"\n[{status}] Test {i}: {preview}")
    print(f"       Rating: {rating:.0f}*  |  Expected: {expected.upper()}  |  "
          f"Got: {predicted.upper()}  |  Fake%: {fake_prob:.1%}")

print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{passed+failed} passed  |  "
      f"{failed} failed  |  {passed/(passed+failed)*100:.0f}% pass rate")
print("=" * 70)

# Summary by category
fake_tests = [(t, r, e) for t, r, e in test_cases if e == "fake"]
real_tests = [(t, r, e) for t, r, e in test_cases if e == "real"]

fake_caught = sum(1 for t, r, e in fake_tests if predict(t, r)[0] == 1)
real_kept = sum(1 for t, r, e in real_tests if predict(t, r)[0] == 0)

print(f"\nFake reviews caught : {fake_caught}/{len(fake_tests)} ({fake_caught/len(fake_tests)*100:.0f}%)")
print(f"Real reviews kept   : {real_kept}/{len(real_tests)} ({real_kept/len(real_tests)*100:.0f}%)")
