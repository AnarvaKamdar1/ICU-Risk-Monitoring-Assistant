"""Reference-note retrieval from MTSamples (TF-IDF), the clinical analogue of the SOC MITRE lookup."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

VAR_TERMS = {
    "HR": "tachycardia heart rate", "MAP": "hypotension blood pressure", "NIMAP": "hypotension blood pressure",
    "SysABP": "hypotension systolic", "NISysABP": "hypotension systolic", "DiasABP": "hypotension", "NIDiasABP": "hypotension",
    "RespRate": "tachypnea respiratory rate shortness of breath", "SaO2": "hypoxia oxygen saturation",
    "PaO2": "hypoxemia arterial blood gas", "FiO2": "oxygen requirement", "PaCO2": "hypercapnia",
    "Lactate": "lactic acidosis sepsis shock", "GCS": "altered mental status", "Temp": "fever",
    "WBC": "leukocytosis infection", "Creatinine": "acute kidney injury renal failure", "BUN": "renal failure",
    "Urine": "oliguria urine output", "HCO3": "metabolic acidosis", "pH": "acidosis", "Platelets": "thrombocytopenia",
    "MechVent": "mechanical ventilation intubation", "Bilirubin": "liver dysfunction", "Glucose": "hyperglycemia",
    "K": "potassium", "Na": "sodium", "HCT": "anemia hematocrit", "Albumin": "albumin",
}
DX_TERMS = {
    "Sepsis": "sepsis septic shock infection", "Respiratory": "respiratory failure pneumonia",
    "Cardiovascular": "cardiac heart failure", "Neurological": "neurologic stroke", "Trauma": "trauma injury",
    "Gastrointestinal": "gastrointestinal bleeding", "Metabolic": "metabolic diabetic ketoacidosis",
    "Genitourinary": "renal urinary", "Hematological": "hematologic", "Musculoskeletal/Skin": "cellulitis",
}


class NoteRetriever:
    def __init__(self, csv_path, max_chars=6000):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["transcription"]).drop_duplicates(subset=["transcription"]).reset_index(drop=True)
        for c in ["description", "medical_specialty", "sample_name", "keywords"]:
            df[c] = df[c].fillna("").astype(str).str.strip() if c in df else ""
        df["text"] = (df.sample_name + ". " + df.description + ". " + df.keywords + ". " +
                      df.transcription.astype(str).str[:max_chars])
        self.df = df
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000)
        self.X = self.vec.fit_transform(df.text)
        self.used = set()

    def build_query(self, dx_group, imaging, top_vars):
        q = [DX_TERMS.get(str(dx_group), str(dx_group)), "intensive care unit"]
        if imaging.get("brixia_total", 0) > 0:
            q.append("pneumonia lung opacity infiltrate consolidation chest x-ray")
            if imaging.get("bilateral"):
                q.append("bilateral infiltrates")
        for v in top_vars:
            q.append(VAR_TERMS.get(v, v))
        return " ".join(q)

    def retrieve(self, query, k=2, excerpt_chars=450):
        s = (self.X @ self.vec.transform([query]).T).toarray().ravel()
        order = [i for i in np.argsort(-s) if i not in self.used][:k]
        terms = set(re.findall(r"[a-z]{4,}", query.lower()))
        out = []
        for i in order:
            self.used.add(i)
            r = self.df.iloc[i]
            sents = re.split(r"(?<=[.!?])\s+|,\s*(?=[A-Z ]{4,}:)", str(r.transcription))
            ranked = sorted(sents, key=lambda x: -len(terms & set(re.findall(r"[a-z]{4,}", x.lower()))))
            excerpt, n = [], 0
            for sent in ranked[:4]:
                if n + len(sent) > excerpt_chars:
                    continue
                excerpt.append(sent.strip()); n += len(sent)
            out.append({"note_id": int(i), "specialty": r.medical_specialty, "title": r.sample_name,
                        "score": round(float(s[i]), 3), "excerpt": " ... ".join(excerpt)[:excerpt_chars]})
        return out
