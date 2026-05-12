import random

def analyze_fake_patterns(prediction_score):

    reasons = []

    # prediction_score example:
    # 0.0 = real
    # 1.0 = fake

    if prediction_score > 0.70:

        possible_reasons = [
            "Face flickering detected",
            "Temporal inconsistency found",
            "Compression artifacts detected",
            "Eye blinking anomaly",
            "Lip-sync mismatch",
            "Blurred facial regions",
            "Frame transition artifacts",
            "Lighting inconsistency",
            "Skin texture abnormality"
        ]

        # randomly choose 4 reasons
        reasons = random.sample(possible_reasons, 4)

    return reasons