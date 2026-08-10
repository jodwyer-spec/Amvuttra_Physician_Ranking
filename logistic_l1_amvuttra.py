import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
import re  
import ast  
import warnings  
warnings.filterwarnings("ignore")

# ============================================================  
# CONFIGURATION  
# ============================================================

INPUT_FILE = "physicians.csv"

NOISE_PENALTY = 10.0

DRUG_TARGET_TERMS = {  
    "cardiac amyloidosis": 50,  
    "cardiac amyloidosis attr": 50,  
    "cardiac amyloidosis al": 50,  
    "amyloid cardiomyopathy": 50,  
    "amyloidosis": 50,  
    "hereditary transthyretin cardiac amyloidosis": 50,  
    "wild type transthyretin attr amyloidosis": 50,  
    "light chain al cardiac amyloidosis": 50,  
    "cardiac sarcoidosis": 50,  
    "heart failure": 30,  
    "advanced heart failure": 30,  
    "restrictive cardiomyopathy": 30,  
    "cardiomyopathy": 30,  
    "dilated cardiomyopathy": 30,  
    "hypertrophic cardiomyopathy": 30,  
    "ventricular dysfunction": 30,  
    "myocarditis": 30,  
    "takotsubo cardiomyopathy": 30,  
    "congestive heart failure": 30,  
    "cardiogenetics": 30,  
    "coronary artery disease": 10,  
    "atrial fibrillation": 10,  
    "hypertension": 10,  
    "valve disorders": 10,  
    "preventive cardiology": 10,  
    "sports cardiology": 10,  
    "cardiology": 5,  
}

SYNONYM_MAP = {  
    "cardiac amyloidosis (attr)": "cardiac amyloidosis attr",  
    "cardiac amyloidosis (al)": "cardiac amyloidosis al",  
    "cardiacamyloidosis (al)": "cardiac amyloidosis al",  
    "cardiacamyloidosis (attr)": "cardiac amyloidosis attr",  
    "light chain (al) cardiac amyloidosis": "light chain al cardiac amyloidosis",  
    "wild-type transthyretin (attr) amyloidosis": "wild type transthyretin attr amyloidosis",  
    "hereditary transthyretin cardiac amyloidosis": "hereditary transthyretin cardiac amyloidosis",  
    "advanced heart failure cardiologist": "advanced heart failure",  
    "congestive heart failure": "congestive heart failure",  
    "cardiomyopathy, dilated": "dilated cardiomyopathy",  
    "coronary arteriosclerosis": "coronary artery disease",  
}

# ============================================================  
# TEXT PROCESSING  
# ============================================================

def clean_text(value):  
    if pd.isna(value):  
        return ""  
    return str(value).lower().strip()

def normalize_term(term):  
    term = clean_text(term)  
    mapped = SYNONYM_MAP.get(term, None)  
    if mapped:  
        return mapped  
    stripped = re.sub(r"[^a-z0-9\s]", " ", term)  
    stripped = re.sub(r"\s+", " ", stripped).strip()  
    mapped = SYNONYM_MAP.get(stripped, None)  
    if mapped:  
        return mapped  
    return stripped

def parse_cfts(value):  
    if pd.isna(value):  
        return []  
    value = str(value).strip()  
    try:  
        parsed = ast.literal_eval(value)  
        if isinstance(parsed, list):  
            return [normalize_term(t) for t in parsed if clean_text(t)]  
    except Exception:  
        pass  
    return [normalize_term(t) for t in re.split(r"[;,|]", value) if clean_text(t)]

def get_provider_terms(row):  
    terms = []  
    spec = normalize_term(row.get("specialty", ""))  
    if spec:  
        terms.append(spec)  
    subspec_raw = str(row.get("subspecialty", ""))  
    subspec = normalize_term(subspec_raw)  
    if subspec:  
        terms.append(subspec)  
        if "-" in subspec_raw:  
            for part in subspec_raw.split("-"):  
                cleaned = normalize_term(part)  
                if cleaned:  
                    terms.append(cleaned)  
    terms.extend(parse_cfts(row.get("clinical_focus_terms", "")))  
    return terms

def match_terms(provider_terms, drug_terms):  
    unique = list(set(provider_terms))  
    matched = {}  
    unmatched = []  
    for term in unique:  
        if term in drug_terms:  
            matched[term] = drug_terms[term]  
        else:  
            unmatched.append(term)  
    return matched, unmatched

# ============================================================  
# LOGISTIC REGRESSION (L1)  
# ============================================================

def run_logistic_l1(providers, drug_terms):  
    vocab = sorted(drug_terms.keys())  
    vocab_idx = {t: i for i, t in enumerate(vocab)}  
    n_features = len(vocab) + 2

    X = np.zeros((len(providers), n_features))  
    for i, p in enumerate(providers):  
        matched = p["matched_terms_dict"]  
        for term, weight in matched.items():  
            if term in vocab_idx:  
                X[i, vocab_idx[term]] = weight  
        X[i, -2] = p["specificity"]  
        X[i, -1] = p["unmatched_count"]

    y = np.zeros(len(providers))  
    for i, p in enumerate(providers):  
        weighted_sum = sum(p["matched_terms_dict"].values())  
        if p["specificity"] >= 0.8 and weighted_sum >= 50:  
            y[i] = 1

    if y.sum() == 0 or y.sum() == len(y):  
        y[0] = 1  
        y[-1] = 0

    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(  
        penalty="l1",  
        solver="saga",  
        C=1.0,  
        max_iter=5000,  
        random_state=42,  
    )  
    model.fit(X_scaled, y)

    probs = model.predict_proba(X_scaled)  
    pos_idx = list(model.classes_).index(1)

    results = []  
    for i, p in enumerate(providers):  
        prob = probs[i, pos_idx]  
        n_unmatched = p["unmatched_count"]  
        total = p["total_term_count"]  
        n_matched = len(p["matched_terms_dict"])  
        specificity = n_matched / total if total > 0 else 0.0

        prob_boosted = prob * (specificity ** 3)

        penalty = n_unmatched * NOISE_PENALTY  
        positive = prob_boosted * 1000  
        direct_score = max(0.0, positive - penalty)

        results.append({  
            "provider_name": p["provider_name"],  
            "algo_score": prob,  
            "positive": round(positive, 2),  
            "penalty": round(penalty, 2),  
            "direct_score": round(direct_score, 2),  
        })

    return results

# ============================================================  
# MAIN  
# ============================================================

def main():  
    df = pd.read_csv(INPUT_FILE)  
    df.columns = [str(c).strip().lower() for c in df.columns]

    providers = []  
    for _, row in df.iterrows():  
        all_terms = get_provider_terms(row)  
        unique = list(set(all_terms))  
        matched, unmatched = match_terms(all_terms, DRUG_TARGET_TERMS)  
        total = len(unique)  
        n_matched = len(matched)  
        n_unmatched = len(unmatched)  
        specificity = n_matched / total if total > 0 else 0.0

        providers.append({  
            "provider_name": row["provider_name"],  
            "npi": str(row["npi"]),  
            "subspecialty": row.get("subspecialty", ""),  
            "all_terms": all_terms,  
            "matched_terms_dict": matched,  
            "matched_terms_list": sorted(matched.keys()),  
            "unmatched_terms": unmatched,  
            "total_term_count": total,  
            "matched_count": n_matched,  
            "unmatched_count": n_unmatched,  
            "specificity": specificity,  
        })

    algo_results = run_logistic_l1(providers, DRUG_TARGET_TERMS)

    rows = []  
    for i, ar in enumerate(algo_results):  
        p = providers[i]  
        rows.append({  
            "provider_name": p["provider_name"],  
            "subspecialty": p["subspecialty"],  
            "algo_score": ar["algo_score"],  
            "positive": ar["positive"],  
            "penalty": ar["penalty"],  
            "direct_score": ar["direct_score"],  
            "matched": p["matched_count"],  
            "unmatched": p["unmatched_count"],  
            "total": p["total_term_count"],  
            "specificity": round(p["specificity"], 3),  
        })

    rdf = pd.DataFrame(rows)  
    rdf = rdf.sort_values("direct_score", ascending=False).reset_index(drop=True)  
    rdf["rank"] = rdf.index + 1

    mx = rdf["direct_score"].max()  
    rdf["score_norm"] = (rdf["direct_score"] / mx * 100).round(1) if mx > 0 else 0

    print(f"\n  LOGISTIC REGRESSION (L1) — AMVUTTRA — {len(df)} providers")  
    print(f"  Noise penalty: -{NOISE_PENALTY:.0f} per unmatched CFT | Strict matching (exact only)")  
    print(f"  {'='*105}")

    print(f"\n  {'Rk':<4} {'Provider':<35} {'Score':>6} {'Pos':>6} {'-Pen':>6} "  
          f"{'Spec':>5} {'M':>3} {'U':>3}  Subspec")  
    print(f"  {'-'*105}")  
    for _, r in rdf.head(15).iterrows():  
        print(f"  {r['rank']:<4} {r['provider_name']:<35} "  
              f"{r['score_norm']:>6.1f} {r['positive']:>6.0f} {r['penalty']:>6.0f} "  
              f"{r['specificity']:>5.3f} {r['matched']:>3} {r['unmatched']:>3}  "  
              f"{r['subspecialty']}")

    print(f"\n  SCORE BREAKDOWN (Top 5):")  
    print(f"  {'-'*105}")  
    for _, r in rdf.head(5).iterrows():  
        print(f"  {r['provider_name']:<35} "  
              f"Pos:{r['positive']:>7.1f} - Pen:{r['penalty']:>6.1f} "  
              f"= Dir:{r['direct_score']:>7.1f}  "  
              f"({r['matched']}M / {r['unmatched']}U / {r['total']}T)  "  
              f"Spec:{r['specificity']:.3f}")


if __name__ == "__main__":  
    main()  
