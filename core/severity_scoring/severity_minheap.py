import csv
import heapq

# Global dictionary to store per-disease severity weights
SEVERITY_WEIGHTS = {}

def load_severity_weights(csv_path='data/severity_weights.csv'):
    global SEVERITY_WEIGHTS
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            disease = row["Disease"].strip()
            SEVERITY_WEIGHTS[disease] = {
                "pain_level": float(row["pain_level"]),
                "itching_level": float(row["itching_level"]),
                "duration_days": float(row["duration_days"]),
                "area_factor": float(row["area_factor"]),
                "fever_present": float(row["fever_present"]),
                "bleeding": float(row["bleeding"]),
                "spread_rate_factor": float(row["spread_rate_factor"]),
            }

class SeverityScorer:
    def __init__(self):
        self.heap = []  # Heap of (score, disease)
        self.direct_scores = {}  # Optional raw score addition

    def add_raw_score(self, disease, score):
        """Add precomputed score (e.g., from image similarity)"""
        self.direct_scores[disease] = self.direct_scores.get(disease, 0) + score

    def calculate_and_add_score(self, disease, pain_level, itching_level, duration_days, max_duration,
                                 area_factor, fever_present, bleeding, spread_rate_factor):
        """
        Calculates weighted severity using global SEVERITY_WEIGHTS.
        """
        if disease not in SEVERITY_WEIGHTS:
            return

        weights = SEVERITY_WEIGHTS[disease]
        score = (
            weights.get("pain_level", 0) * (pain_level / 10) +
            weights.get("itching_level", 0) * (itching_level / 10) +
            weights.get("duration_days", 0) * (duration_days / max_duration if max_duration else 0) +
            weights.get("area_factor", 0) * area_factor +
            weights.get("fever_present", 0) * int(fever_present) +
            weights.get("bleeding", 0) * int(bleeding) +
            weights.get("spread_rate_factor", 0) * spread_rate_factor
        )

        heapq.heappush(self.heap, (round(score, 4), disease))

    def get_most_severe(self):
        if self.heap:
            return heapq.nlargest(1, self.heap)[0]
        if self.direct_scores:
            return max(self.direct_scores.items(), key=lambda x: x[1])
        return None

    def get_all_ranked(self):
        if self.heap:
            return heapq.nlargest(len(self.heap), self.heap)
        return sorted(self.direct_scores.items(), key=lambda x: x[1], reverse=True)
