import re

# Psychological Tactic Keywords
TACTICS = {
    "Fear": ["suspend", "block", "cancel", "disconnection", "unauthorized", "suspension", "notice", "arrest", "court"],
    "Urgency": ["immediate", "instantly", "now", "today", "expire", "hurry", "fast", "quickly", "limited", "hours"],
    "Greed": ["won", "gift", "card", "lucky", "draw", "win", "cashback", "reward", "congratulations", "double", "money"],
    "Authority": ["bank", "sbi", "aadhaar", "official", "government", "subsidy", "it", "tax", "income", "regulation"],
    "Trust Hijack": ["netflix", "amazon", "zomato", "swiggy", "friend", "joh", "payment", "confirmed", "verified"]
}

# Weighted keywords for scoring
WEIGHTS = {
    "Fear": 15,
    "Urgency": 10,
    "Greed": 15,
    "Authority": 10,
    "Trust Hijack": 5
}

def get_psychological_signals(text):
    text = text.lower()
    detected = {}
    score = 0
    
    for tactic, keywords in TACTICS.items():
        found = [kw for kw in keywords if re.search(rf'\b{kw}\b', text)]
        if found:
            detected[tactic] = found
            score += WEIGHTS[tactic]
            
    # Normalize score (max weight is roughly 55, but we can cap it)
    normalized_score = min(30, score) # Rule based engine contributes up to 30 points as per requirement
    
    # Identify primary trigger
    primary_trigger = "None"
    if detected:
        primary_trigger = max(detected.keys(), key=lambda x: WEIGHTS[x])
        
    return {
        "rule_score": normalized_score,
        "primary_trigger": primary_trigger,
        "signals": detected
    }

if __name__ == "__main__":
    test_msgs = [
        "Your KYC will expire today. Update immediately to avoid account suspension.",
        "Congratulations! You won ₹10,00,000 in lucky draw."
    ]
    for msg in test_msgs:
        print(f"\nMessage: {msg}")
        print(get_psychological_signals(msg))
