import csv

treatment_map = {
    "eczema": {
        "severity": "Moderate",
        "recommended_treatment": [
            "Topical corticosteroids",
            "Moisturizers",
            "Antihistamines"
        ],
        "icd_10": "L20.9"
    },
    "warts, molluscum, and other viral infections": {
        "severity": "Mild",
        "recommended_treatment": [
            "Cryotherapy",
            "Salicylic acid",
            "Topical antivirals"
        ],
        "icd_10": "B07.9"
    },
    "melanoma": {
        "severity": "Severe",
        "recommended_treatment": [
            "Surgical excision",
            "Immunotherapy",
            "Targeted therapy"
        ],
        "icd_10": "C43.9"
    },
    "atopic dermatitis": {
        "severity": "Moderate",
        "recommended_treatment": [
            "Topical steroids",
            "Emollients",
            "Immunomodulators"
        ],
        "icd_10": "L20.0"
    },
    "basal cell carcinoma (bcc)": {
        "severity": "Severe",
        "recommended_treatment": [
            "Mohs surgery",
            "Radiation therapy",
            "Topical chemo"
        ],
        "icd_10": "C44.91"
    },
    "melanocytic nevi (nv)": {
        "severity": "Mild",
        "recommended_treatment": [
            "Observation",
            "Excision if suspicious or for cosmetic reasons"
        ],
        "icd_10": "D22.9"
    },
    "benign keratosis-like lesions (bkl)": {
        "severity": "Mild",
        "recommended_treatment": [
            "Cryotherapy",
            "Curettage",
            "Laser removal"
        ],
        "icd_10": "L82.1"
    },
    "psoriasis, lichen planus, and related diseases": {
        "severity": "Moderate to Severe",
        "recommended_treatment": [
            "Topical corticosteroids",
            "Phototherapy",
            "Biologics"
        ],
        "icd_10": "L40.9"
    },
    "seborrheic keratoses and other benign tumors": {
        "severity": "Mild",
        "recommended_treatment": [
            "Cryosurgery",
            "Curettage",
            "No treatment if asymptomatic"
        ],
        "icd_10": "L82.0"
    },
    "tinea, ringworm, candidiasis, and other fungal infections": {
        "severity": "Moderate",
        "recommended_treatment": [
            "Topical antifungals",
            "Oral antifungals in severe cases"
        ],
        "icd_10": "B35.9"
    }
}


class TreatmentMap:
    def __init__(self, csv_path):
        self.map = {}
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.map[row['Disease'].lower()] = row['Recommended Treatment']

    def get_treatment(self, disease):
        if not isinstance(disease, str):
            disease = str(disease)
        return self.map.get(disease.lower(), "Treatment info not available.")
