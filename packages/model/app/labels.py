ID2LABEL = {
    0: "neutral",
    1: "sadness_grief",
    2: "joy_amusement_excitement_optimism",
    3: "anger_annoyance_disapproval_disgust",
    4: "desire",
    5: "fear_nervousness",
    6: "love",
}

LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

DISPLAY_LABELS = {
    0: "Normal (neutral)",
    1: "Sadness (sadness, grief)",
    2: "Joy (joy, amusement, excitement, optimism)",
    3: "Hate / Anger (anger, annoyance, disapproval, disgust)",
    4: "Sexual / Desire (desire)",
    5: "Fear / Anxiety (fear, nervousness)",
    6: "Love / ভালোবাসা (love)",
}

NUM_LABELS = len(ID2LABEL)
MAX_LENGTH = 128
BASE_MODEL = "roberta-base"
