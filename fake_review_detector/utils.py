"""
Shared utilities for Fake Review Detector.
Used by both train.py and app.py to ensure consistent preprocessing.
"""

import re
import numpy as np


def _normalize_quotes(text):
    """Replace smart/curly quotes with straight ASCII equivalents.
    ChatGPT, Word, and many apps use Unicode quotes that break phrase matching.
    """
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # ' '
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    text = text.replace("\u2032", "'").replace("\u2033", '"')   # prime marks
    text = text.replace("\u00b4", "'").replace("\u0060", "'")   # acute/grave accent
    return text


def clean_text(text):
    """Clean text for TF-IDF vectorization."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)          # remove HTML tags
    text = re.sub(r'http\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)       # keep only letters + spaces
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    return text


def extract_behavioural_features(text, rating=3.0):
    """
    Extract 29 behavioural / stylistic features from the RAW review text.
    These complement TF-IDF by capturing writing style, not just word content.

    Features cover: review length, vocabulary richness, punctuation style,
    sentence structure, sentiment cues, excessive positivity, repeated
    adjectives, exclamation abuse, generic wording, lack of specific
    details, sentiment inconsistency, persuasive/commanding language,
    and AI marketing buzzwords.

    Returns a list of 29 floats.
    """
    from collections import Counter

    t = _normalize_quotes(str(text))
    words = t.split()
    num_words = len(words)
    tl = t.lower()
    lower_words = tl.split()

    # ── Basic length features ───────────────────────
    word_count = num_words
    char_count = len(t)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    max_word_len = max((len(w) for w in words), default=0)

    # ── Vocabulary richness ─────────────────────────
    unique_ratio = len(set(lower_words)) / max(num_words, 1)
    # Hapax legomena ratio (words that appear exactly once)
    word_freq = Counter(lower_words)
    hapax_ratio = sum(1 for c in word_freq.values() if c == 1) / max(num_words, 1)

    # ── Punctuation / style features ────────────────
    exclaim_count = t.count('!')
    question_count = t.count('?')
    ellipsis_count = t.count('...')
    caps_ratio = sum(1 for c in t if c.isupper()) / max(len(t), 1)
    digit_ratio = sum(1 for c in t if c.isdigit()) / max(len(t), 1)

    # ── Sentence-level features ─────────────────────
    sentences = re.split(r'[.!?]+', t)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    avg_sentence_len = num_words / num_sentences

    # ── Sentiment / hype cues ───────────────────────
    hype_words = [
        'amazing', 'incredible', 'awesome', 'perfect', 'excellent',
        'fantastic', 'wonderful', 'love', 'best', 'great',
        'must buy', 'highly recommend', 'changed my life',
        'love love', 'absolutely', 'definitely',
        # Expanded: superlatives & AI-style praise words
        'unbelievable', 'outstanding', 'remarkable', 'phenomenal',
        'stunning', 'flawless', 'exceptional', 'superb', 'brilliant',
        'mind blowing', 'game changer', 'life changing', 'top notch',
        'worth every penny', 'hands down', 'no brainer',
    ]
    negative_words = [
        'terrible', 'horrible', 'worst', 'awful', 'waste',
        'garbage', 'trash', 'scam', 'broke', 'useless',
        'disappointing', 'regret', 'refund', 'return',
    ]
    hype_count = sum(1 for h in hype_words if h in tl)
    negative_count = sum(1 for n in negative_words if n in tl)

    # ── Rating-related features ─────────────────────
    is_extreme = 1 if rating in (1.0, 5.0) else 0
    hype_x_extreme = hype_count * is_extreme

    # ── NEW: Excessive positivity ───────────────────
    positivity_ratio = hype_count / max(num_words, 1)
    is_excessive_positivity = 1 if (positivity_ratio > 0.15 and rating == 5.0) else 0

    # ── NEW: Repeated adjectives ────────────────────
    # Common adjectives used in reviews
    _adjectives = {
        'good', 'great', 'best', 'perfect', 'amazing', 'awesome',
        'excellent', 'fantastic', 'wonderful', 'incredible', 'nice',
        'terrible', 'horrible', 'worst', 'awful', 'bad', 'poor',
        'beautiful', 'lovely', 'superb', 'outstanding', 'brilliant',
    }
    adj_counts = {w: word_freq[w] for w in _adjectives if word_freq.get(w, 0) > 1}
    repeated_adjectives = sum(adj_counts.values()) - len(adj_counts)  # excess occurrences

    # Consecutive identical words (e.g. "great great great")
    consecutive_repeats = sum(
        1 for w1, w2 in zip(lower_words[:-1], lower_words[1:]) if w1 == w2
    )

    # ── NEW: Too many exclamation marks ─────────────
    exclaim_ratio = exclaim_count / max(num_words, 1)
    has_excessive_exclaims = 1 if exclaim_count >= 3 else 0

    # ── NEW: Generic wording ────────────────────────
    _generic_words = {
        'product', 'item', 'good', 'great', 'nice', 'okay', 'fine',
        'buy', 'bought', 'purchase', 'recommend', 'review', 'seller',
        'shipping', 'delivery', 'box', 'package',
    }
    generic_ratio = sum(1 for w in lower_words if w in _generic_words) / max(num_words, 1)

    # ── NEW: Lack of specific details ───────────────
    _spec_terms = [
        'inch', 'inches', 'mm', 'cm', 'lbs', 'oz', 'gram', 'grams',
        'size', 'color', 'colour', 'fit', 'battery', 'stitching',
        'material', 'watt', 'watts', 'volt', 'pixel', 'resolution',
        'model', 'version', 'diameter', 'weight', 'height', 'width',
    ]
    has_specifics = 1 if (
        digit_ratio > 0 or any(term in tl for term in _spec_terms)
    ) else 0
    lack_of_details = 1 - has_specifics

    # ── NEW: Sentiment inconsistency ────────────────
    # Rating mapped to [-1, 1]: 1★ → -1.0, 3★ → 0.0, 5★ → 1.0
    rating_sentiment = (rating - 3.0) / 2.0
    text_sentiment = 0.0
    if (hype_count + negative_count) > 0:
        text_sentiment = (hype_count - negative_count) / (hype_count + negative_count)
    sentiment_inconsistency = (
        abs(rating_sentiment - text_sentiment)
        if (hype_count + negative_count) > 0
        else 0.0
    )

    # ── NEW: AI Marketing Buzzwords ─────────────────
    _ai_buzzwords = [
        "beyond satisfied", "buying experience", "premium", "top-class", 
        "value for money", "unbeatable", "stands above the rest", 
        "best possible experience", "performs perfectly", "without any issues",
        "highly impressed", "exceeded my expectations", "can't recommend enough",
        "cannot recommend enough", "seamless", "top-notch", "impeccable",
        "pleasantly surprised", "highly recommend this product",
        "little hesitant", "exactly what was promised", "customer satisfaction",
        "carefully designed", "extremely happy with my purchase", 
        "buy from this brand again", "simply outstanding"
    ]
    ai_buzzword_score = sum(1 for bw in _ai_buzzwords if bw in tl)

    # ── NEW: Persuasive / commanding language ───────
    _persuasion_phrases = [
        "don't think twice", "dont think twice",
        "don't hesitate", "dont hesitate",
        "you won't regret", "you wont regret",
        "trust me", "believe me",
        "buy it now", "get this now", "you need this",
        "don't miss", "dont miss", "what are you waiting for",
        "just buy", "go for it", "stop looking",
        "look no further", "do yourself a favor",
        "take my word", "worth every cent",
        "no questions asked", "period.",
        "enough said", "nuff said", "say no more",
    ]
    persuasion_score = sum(1 for p in _persuasion_phrases if p in tl)

    return [
        word_count,
        char_count,
        avg_word_len,
        max_word_len,
        unique_ratio,
        hapax_ratio,
        exclaim_count,
        question_count,
        ellipsis_count,
        caps_ratio,
        digit_ratio,
        num_sentences,
        avg_sentence_len,
        hype_count,
        negative_count,
        rating,
        is_extreme,
        hype_x_extreme,
        # --- New features ---
        positivity_ratio,
        is_excessive_positivity,
        repeated_adjectives,
        consecutive_repeats,
        exclaim_ratio,
        has_excessive_exclaims,
        generic_ratio,
        lack_of_details,
        sentiment_inconsistency,
        persuasion_score,
        ai_buzzword_score,
    ]


# Feature names for interpretability / debugging
FEATURE_NAMES = [
    'word_count', 'char_count', 'avg_word_len', 'max_word_len',
    'unique_ratio', 'hapax_ratio',
    'exclaim_count', 'question_count', 'ellipsis_count',
    'caps_ratio', 'digit_ratio',
    'num_sentences', 'avg_sentence_len',
    'hype_count', 'negative_count',
    'rating', 'is_extreme', 'hype_x_extreme',
    # --- New features ---
    'positivity_ratio', 'is_excessive_positivity',
    'repeated_adjectives', 'consecutive_repeats',
    'exclaim_ratio', 'has_excessive_exclaims',
    'generic_ratio', 'lack_of_details',
    'sentiment_inconsistency', 'persuasion_score', 'ai_buzzword_score',
]


def compute_heuristic_score(text, rating=3.0):
    """
    Computes a heuristic spam/fake score based on writing signals and disclaimers.
    Returns (score, list_of_signals)
    """
    t = _normalize_quotes(str(text)).lower()
    words = t.split()
    num_words = len(words)
    
    score = 0.0
    signals = []
    
    # 1. Incentivized review disclaimers
    disclaimers = [
        "received this product for free",
        "free in exchange for",
        "in exchange for my honest review",
        "free product in exchange",
        "for free in exchange",
        "received this for free",
    ]
    if any(d in t for d in disclaimers):
        score += 0.65
        signals.append("Incentivized review disclaimer")
        
    # 2. Extreme repetition
    unique_ratio = len(set(words)) / max(num_words, 1)
    if unique_ratio < 0.4 and num_words >= 5:
        score += 0.55
        signals.append("Highly repetitive phrasing")
    elif unique_ratio < 0.6 and num_words >= 10:
        score += 0.35
        signals.append("Repetitive vocabulary")
        
    # 3. Excessive exclamation marks
    exclaims = text.count('!')
    if exclaims >= 8:
        score += 0.60
        signals.append("Spammy exclamation usage")
    elif exclaims >= 3:
        score += 0.40
        signals.append("Frequent exclamation marks")
        
    # 4. Extreme CAPS
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.40 and len(text) > 15:
        score += 0.55
        signals.append("Shouting style (excessive CAPS)")
    elif caps_ratio > 0.20 and len(text) > 15:
        score += 0.30
        signals.append("High CAPS usage")
        
    # 5. Overly short reviews (low-effort / spam)
    if num_words <= 2:
        score += 0.55
        signals.append("Extremely short review")
    elif num_words <= 5 and rating == 5.0:
        score += 0.45
        signals.append("Short 5-star review")
    elif num_words <= 12 and rating == 5.0:
        score += 0.35
        signals.append("Short 5-star review")
            
    # 6. Sentiment/Hype words overload
    hype_phrases = [
        "best product ever", "changed my life", "must buy", "love love",
        "buy this right now", "amazing product", "best purchase ever",
        "absolutely perfect", "highly recommend", "five stars", "awesome awesome",
        "perfect perfect", "everyone should buy", "best ever", "love it so much",
        "recommend it to all", "recommend to everyone", "buy this right", "buy now",
        # Expanded: AI-style praise
        "worth every penny", "game changer", "life changing", "top notch",
        "hands down", "no brainer", "mind blowing",
    ]
    hype_matches = sum(1 for hp in hype_phrases if hp in t)
    
    hype_words = [
        "amazing", "incredible", "awesome", "perfect", "fantastic",
        "wonderful", "great", "love",
        # Expanded
        "unbelievable", "phenomenal", "outstanding", "remarkable",
        "stunning", "flawless", "exceptional", "superb", "brilliant",
    ]
    word_matches = sum(1 for hw in hype_words if hw in words)
    
    total_hype = hype_matches * 2 + word_matches
    if rating == 5.0 and total_hype >= 5:
        score += 0.55
        signals.append("Extreme hype overload")
    elif rating == 5.0 and total_hype >= 3:
        score += 0.40
        signals.append("High hype density")
    elif rating == 5.0 and total_hype >= 2:
        score += 0.25
        signals.append("Moderate hype density")

    # 7. Persuasive / commanding language (AI-generated review patterns)
    persuasion_phrases = [
        "don't think twice", "dont think twice",
        "don't hesitate", "dont hesitate",
        "you won't regret", "you wont regret",
        "trust me", "believe me",
        "buy it now", "get this now", "you need this",
        "don't miss", "dont miss", "what are you waiting for",
        "just buy", "go for it", "stop looking",
        "look no further", "do yourself a favor",
        "take my word", "worth every cent",
        "no questions asked", "enough said", "say no more",
    ]
    persuasion_hits = sum(1 for p in persuasion_phrases if p in t)
    if persuasion_hits >= 2:
        score += 0.55
        signals.append("Pushy / commanding language")
    elif persuasion_hits >= 1:
        score += 0.40
        signals.append("Persuasive language detected")

    # 8. Vague short praise with no specifics (classic AI-generated pattern)
    has_digits = any(c.isdigit() for c in text)
    spec_terms = [
        'inch', 'mm', 'cm', 'lbs', 'oz', 'size', 'color', 'colour',
        'fit', 'battery', 'stitching', 'material', 'watt', 'pixel'
    ]
    has_specs = has_digits or any(st in t for st in spec_terms)
    hype_or_persuasion = total_hype + persuasion_hits
    
    # Generic entity references vs specific product names
    generic_entities = sum(1 for e in ["this product", "this item", "this brand", "this company"] if e in t)
    
    if num_words <= 15 and not has_specs and hype_or_persuasion >= 1:
        score += 0.35
        signals.append("Vague short praise with no details")
        
    # 9. Vague verbose praise (ChatGPT style)
    # Long review but completely devoid of specific details and uses AI marketing buzzwords
    ai_buzzwords = [
        "beyond satisfied", "buying experience", "premium", "top-class", 
        "value for money", "unbeatable", "stands above the rest", 
        "best possible experience", "performs perfectly", "without any issues",
        "highly impressed", "exceeded my expectations", "can't recommend enough",
        "cannot recommend enough", "seamless", "top-notch",
        "little hesitant", "exactly what was promised", "customer satisfaction",
        "carefully designed", "extremely happy with my purchase", 
        "buy from this brand again", "simply outstanding"
    ]
    ai_buzz_hits = sum(1 for bw in ai_buzzwords if bw in t)
    
    if rating == 5.0 and num_words > 15:
        if ai_buzz_hits >= 3 or (ai_buzz_hits >= 2 and generic_entities >= 2):
            score += 0.85
            signals.append("Heavy use of AI-style marketing buzzwords")
        elif ai_buzz_hits >= 2 and not has_specs:
            score += 0.65
            signals.append("Vague AI-style marketing praise")
        elif (ai_buzz_hits == 1 or hype_or_persuasion >= 2) and not has_specs:
            score += 0.45
            signals.append("Verbose praise with no specific details")
            
    return min(score, 1.0), signals


def hybrid_predict(text, rating, model, tfidf, scaler):
    """
    Combines the machine learning model prediction with heuristic indicators
    to output a robust, calibrated fake review probability and label.
    """
    import scipy.sparse as sp
    cleaned = clean_text(text)
    X_tfidf = tfidf.transform([cleaned])
    behav = extract_behavioural_features(text, rating)
    X_behav = scaler.transform([behav])
    X = sp.hstack([X_tfidf, sp.csr_matrix(X_behav, dtype=np.float32)])
    
    # 1. ML probability
    ml_fake_prob = float(model.predict_proba(X)[0][1])
    
    # 2. Heuristic indicators
    h_score, signals = compute_heuristic_score(text, rating)
    
    # 3. Hybrid combination logic
    if h_score >= 0.35:
        # High confidence spam signals override/boost fake prob
        final_fake_prob = max(ml_fake_prob, h_score)
    elif h_score >= 0.20:
        # Medium confidence spam signals blend with ML
        final_fake_prob = 0.6 * ml_fake_prob + 0.4 * h_score
    else:
        # Low/no spam signals: rely on ML
        final_fake_prob = ml_fake_prob
        
    # Calibration to prevent false positives on genuine descriptive reviews
    if len(signals) == 0 and len(text.split()) >= 15 and rating in (2.0, 3.0, 4.0):
        final_fake_prob = min(final_fake_prob, 0.40)
        
    predicted_label = 1 if final_fake_prob >= 0.50 else 0
    return predicted_label, final_fake_prob, 1.0 - final_fake_prob


def highlight_review(text):
    """
    Returns an HTML string where AI buzzwords are highlighted in rose red 
    and specific product details/specs are highlighted in cyan.
    """
    import re
    highlighted = str(text)
    
    # Define lists to match heuristic logic
    ai_buzzwords = [
        "beyond satisfied", "buying experience", "premium", "top-class", 
        "value for money", "unbeatable", "stands above the rest", 
        "best possible experience", "performs perfectly", "without any issues",
        "highly impressed", "exceeded my expectations", "can't recommend enough",
        "cannot recommend enough", "seamless", "top-notch", "impeccable",
        "pleasantly surprised", "highly recommend this product",
        "little hesitant", "exactly what was promised", "customer satisfaction",
        "carefully designed", "extremely happy with my purchase", 
        "buy from this brand again", "simply outstanding",
        "don't think twice", "dont think twice", "don't hesitate", "dont hesitate",
        "you won't regret", "trust me", "buy it now", "get this now", "you need this", 
        "don't miss", "what are you waiting for", "just buy", "go for it", 
        "stop looking", "look no further", "do yourself a favor", "take my word", 
        "worth every cent", "no questions asked", "say no more"
    ]
    
    spec_terms = [
        'inch', 'mm', 'cm', 'lbs', 'oz', 'size', 'color', 'colour',
        'fit', 'battery', 'stitching', 'material', 'watt', 'pixel',
        'weight', 'screen', 'cable', 'plastic', 'metal', 'glass', 'fabric', 
        'scent', 'smell', 'taste', 'ingredient', 'button', 'zipper', 'pocket', 
        'strap', 'handle', 'box', 'packaging', 'plug', 'port', 'usb', 'bluetooth', 
        'wifi', 'app', 'update'
    ]
    
    # Sort by length descending so longer phrases match first (prevent partial overlap issues)
    ai_buzzwords.sort(key=len, reverse=True)
    spec_terms.sort(key=len, reverse=True)
    
    def replace_keep_case(word, replacement_template, text):
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        return pattern.sub(lambda match: replacement_template.format(match.group(0)), text)

    # Highlight AI Buzzwords in Rose / Red
    for bw in ai_buzzwords:
        template = '<span style="background-color: rgba(244, 63, 94, 0.4); color: #fff; padding: 2px 6px; border-radius: 6px; border: 1px solid #f43f5e;" title="AI Buzzword / Persuasive Language">{}</span>'
        highlighted = replace_keep_case(bw, template, highlighted)
        
    # Highlight Specific Terms in Cyan / Green
    for term in spec_terms:
        # Match whole words only for short specific terms to avoid weird overlaps
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        template = '<span style="background-color: rgba(6, 182, 212, 0.4); color: #fff; padding: 2px 6px; border-radius: 6px; border: 1px solid #06b6d4;" title="Specific Detail">{}</span>'
        highlighted = pattern.sub(lambda match: template.format(match.group(0)), highlighted)

    # Replace newlines with <br> for HTML rendering
    highlighted = highlighted.replace('\\n', '<br>').replace('\n', '<br>')
    return highlighted
