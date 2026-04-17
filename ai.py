import numpy as np

def predict_attendance(records):
    if len(records) < 3:
        return "Not enough data"

    y = np.array(records)
    trend = np.polyfit(range(len(y)), y, 1)[0]

    if trend < 0:
        return "Declining"
    elif trend > 0:
        return "Improving"
    return "Stable"

def risk_level(count, total):
    if total == 0:
        return "Unknown"

    percent = (count / total) * 100

    if percent < 50:
        return "High"
    elif percent < 75:
        return "Medium"
    return "Low"

def leaderboard(data):
    return sorted(data.items(), key=lambda x: x[1], reverse=True)[:5]