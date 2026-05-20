🚀 𝗨𝗻𝗺𝗮𝘀𝗸𝗶𝗻𝗴 𝗘-𝗖𝗼𝗺𝗺𝗲𝗿𝗰𝗲 𝗦𝗽𝗮𝗺

### 𝗕𝘂𝗶𝗹𝗱𝗶𝗻𝗴 𝗮 𝗛𝘆𝗯𝗿𝗶𝗱 𝗙𝗮𝗸𝗲 𝗥𝗲𝘃𝗶𝗲𝘄 𝗗𝗲𝘁𝗲𝗰𝘁𝗼𝗿

In today’s AI-driven internet, generating highly convincing fake reviews has become easier than ever.
Large Language Models can now create realistic product feedback in seconds — making traditional spam detection systems increasingly ineffective.

To explore this challenge, I built **Fake Review Detector** 🕵️‍♂️
An Applied Machine Learning project trained on **40,000 Amazon reviews**, designed to detect deceptive reviews by analyzing not just *what* is written, but *how* it is written.

╔════════════════════════════════════════════╗
🔗 𝗟𝗶𝘃𝗲 𝗗𝗲𝗺𝗼
https://fake-review-detectors.streamlit.app/

🔗 𝗚𝗶𝘁𝗛𝘂𝗯 𝗥𝗲𝗽𝗼
[fake-review-detector](https://github.com/swastiksinha1/fake-review-detector?utm_source=chatgpt.com)

📄 𝗟𝗶𝗰𝗲𝗻𝘀𝗲
MIT License
╚════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 𝗪𝗵𝘆 𝗧𝗿𝗮𝗱𝗶𝘁𝗶𝗼𝗻𝗮𝗹 𝗡𝗟𝗣 𝗜𝘀𝗻’𝘁 𝗘𝗻𝗼𝘂𝗴𝗵
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Most fake-review systems rely heavily on:
• TF-IDF
• Bag-of-Words
• Basic text classification

However, modern AI-generated reviews often use highly natural vocabulary and realistic sentence structures.

Instead of only analyzing **what is written**, my system analyzes **how it is written**.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ 𝗛𝘆𝗯𝗿𝗶𝗱 𝗣𝗿𝗲𝗱𝗶𝗰𝘁𝗶𝗼𝗻 𝗘𝗻𝗴𝗶𝗻𝗲
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```text
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Analysis Layer               │ Purpose                                      │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ TF-IDF Vectorization         │ Detects spam vocabulary & word patterns      │
│ Behavioral Feature Engine    │ Extracts 29 stylistic writing features       │
│ Heuristic Intelligence Layer │ Flags repetitive & AI-like persuasive text   │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

The final prediction combines:

✔ Machine Learning probabilities
✔ Behavioral analysis
✔ Rule-based confidence scoring

This hybrid architecture significantly reduces false positives on genuinely enthusiastic reviews.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 𝗗𝗲𝘁𝗲𝗰𝘁𝗶𝗻𝗴 𝘁𝗵𝗲 “𝗔𝗜 𝗙𝗶𝗻𝗴𝗲𝗿𝗽𝗿𝗶𝗻𝘁”
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

One of the most interesting parts of the project was identifying patterns commonly found in AI-generated reviews.

### Custom Feature Engineering Included:

➤ **Excessive Positivity Ratios**
Detects unnatural densities of hype words.

➤ **Lack of Product Specificity**
Flags verbose reviews lacking concrete product specifications.

➤ **Persuasive Language Overload**
Detects phrases like:
• “Don’t think twice”
• “Look no further”
• “Best product ever”

➤ **Sentiment Contradiction Detection**
Example:
⭐ 1-star rating with highly positive review text.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 𝗠𝗼𝗱𝗲𝗹 𝗣𝗲𝗿𝗳𝗼𝗿𝗺𝗮𝗻𝗰𝗲
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```text
┌──────────────────────┬──────────┐
│ Metric               │ Score    │
├──────────────────────┼──────────┤
│ Best Model           │ XGBoost  │
│ Accuracy             │ 92.5%    │
│ F1-Score             │ 91.8%    │
│ AUC-ROC              │ 95.0%    │
└──────────────────────┴──────────┘
```

📌 Full model comparisons and validation statistics are available directly inside the application dashboard.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💻 𝗧𝗲𝗰𝗵 𝗦𝘁𝗮𝗰𝗸
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### Machine Learning

• scikit-learn
• pandas
• numpy
• scipy

### Backend & Deployment

• Python
• Streamlit Community Cloud

### Frontend / UI Engineering

• Custom CSS Glassmorphism Interface
• Animated Fluid Backgrounds
• Floating Metric Cards
• Interactive Responsive Design

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 𝗪𝗵𝗮𝘁’𝘀 𝗡𝗲𝘅𝘁?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Moving forward, I plan to explore:

→ Transformer-based embeddings
→ LLM-assisted detection systems
→ Semantic consistency analysis
→ Advanced adversarial spam detection

This project became a fantastic deep dive into:

𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 • 𝗡𝗟𝗣 • 𝗙𝗲𝗮𝘁𝘂𝗿𝗲 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿𝗶𝗻𝗴 • 𝗔𝗜 𝗗𝗲𝗽𝗹𝗼𝘆𝗺𝗲𝗻𝘁 • 𝗙𝘂𝗹𝗹-𝗦𝘁𝗮𝗰𝗸 𝗗𝗲𝘃𝗲𝗹𝗼𝗽𝗺𝗲𝗻𝘁

Would love to hear your feedback, suggestions, or edge-case reviews to test against the model 👇

#MachineLearning #Python #DataScience #NLP #ArtificialIntelligence #AI #Streamlit #CyberSecurity #OpenSource #WebDevelopment #TechInnovation #VITBhopal
