import pandas as pd
import json
import requests
import time
import os
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =================================================================
#                         1. Parameter configuration area
# =================================================================
CONFIG = {
    "INPUT_CSV": "../text/TableS19.csv",
    "OUTPUT_CSV": "../results/Table S19——Pharmacist_Audit.csv",
    "KB_CSV": "../data/DrugBank_Knowledge_Base.csv",
    "PROT_KB": "../data/Prot_KB.json",
    "TEMP_KB_DIR": "../data/guide",
    "MODEL_NAME": "gemma3:12b",
    "OLLAMA_URL": "xxx", # Please enter the AIP provided by ollama

    "EVAL_TOP_N": 1000,
    "SAVE_INTERVAL": 5
}

MCC_MAP = {
    "MCC-1": ["HLP"],
    "MCC-2": ["HLP", "HTN"],
    "MCC-345": ["HLP", "HTN", "FLD"],
    "MCC-6": ["HLP", "HTN", "FLD", "T2D", "HUA", "OBE"],
    "MCC-7": ["HLP", "HTN", "FLD", "T2D", "HUA", "OBE"],
    "MCC-8": ["HLP", "HTN", "FLD", "T2D", "HUA", "OBE"]
}


# =================================================================
#               2. Model performance and stability tracker
# =================================================================
class ReviewerPerformanceTracker:
    def __init__(self):
        self.total_calls = 0
        self.successful_json_parses = 0
        self.total_drugs_processed = 0
        self.drugs_found_in_kb = 0
        self.latencies = []
        self.generated_scores = []

    def log_call(self, latency, success, drugs_in_regimen, kbs_matched, score):
        self.total_calls += 1
        if success:
            self.successful_json_parses += 1
        self.total_drugs_processed += drugs_in_regimen
        self.drugs_found_in_kb += kbs_matched
        self.latencies.append(latency)
        if score is not None:
            try:
                self.generated_scores.append(float(score))
            except (ValueError, TypeError):
                pass

    def print_reviewer_metrics_panel(self):
        json_rate = (self.successful_json_parses / self.total_calls * 100) if self.total_calls > 0 else 0
        grounding_rate = (
                    self.drugs_found_in_kb / self.total_drugs_processed * 100) if self.total_drugs_processed > 0 else 0
        avg_lat = np.mean(self.latencies) if self.latencies else 0

        scores_arr = np.array(self.generated_scores)
        mean_s = np.mean(scores_arr) if len(scores_arr) > 0 else 0
        sd_s = np.std(scores_arr) if len(scores_arr) > 1 else 0
        min_s = np.min(scores_arr) if len(scores_arr) > 0 else 0
        max_s = np.max(scores_arr) if len(scores_arr) > 0 else 0

        print("\n================================================================")
        print("📊      CRITICAL MODEL PERFORMANCE & RIGOR METRICS PANEL        ")
        print("================================================================")
        print(
            f"🔹 1. JSON Structural Compliance Rate:    {json_rate:.2f}% ({self.successful_json_parses}/{self.total_calls})")
        print(
            f"🔹 2. KB Retrieval Grounding Coverage:    {grounding_rate:.2f}% ({self.drugs_found_in_kb}/{self.total_drugs_processed})")
        print(f"🔹 3. Mean Inference Latency per Audit:   {avg_lat:.2f} seconds")
        print(f"🔹 4. EES Score Distribution Profile:")
        print(f"     - Mean Score: {mean_s:.4f}  |  Standard Deviation (SD): {sd_s:.4f}")
        print(f"     - Minimum:    {min_s:.1f}  |  Maximum:                 {max_s:.1f}")
        print("================================================================\n")


# =================================================================
#                        3. Audit core driver class
# =================================================================
class MORPH_Pharmacist_Agent:
    def __init__(self):
        self.tracker = ReviewerPerformanceTracker()

        if not os.path.exists(CONFIG["PROT_KB"]):
            print(f"❌ Warning: Protein knowledge base file not found {CONFIG['PROT_KB']}！")
            self.prot_data = {}
        else:
            with open(CONFIG["PROT_KB"], 'r', encoding='utf-8') as f:
                raw_prot_kb = json.load(f)
                self.prot_data = raw_prot_kb.get('proteins', raw_prot_kb)

        self.kb_dict = {}
        if os.path.exists(CONFIG["KB_CSV"]):
            for enc in ['utf-8-sig', 'gbk', 'utf-8', 'gb18030']:
                try:
                    df_kb = pd.read_csv(CONFIG["KB_CSV"], encoding=enc)

                    for _, row in df_kb.iterrows():
                        drug_name = str(row.get('Drug names', '')).strip().lower()
                        if not drug_name or drug_name in ['nan', 'none', 'null']:
                            continue

                        def get_valid_str(val):
                            v_str = str(val).strip()
                            if v_str == "" or v_str.lower() in ["nan", "n/a", "none", "null"]:
                                return ""
                            return v_str

                        atc = get_valid_str(row.get('ATC Information'))
                        desc = get_valid_str(row.get('Description'))
                        ind = get_valid_str(row.get('Indication'))
                        mech = get_valid_str(row.get('Mechanism of Action'))
                        tox = get_valid_str(row.get('Toxicity'))

                        t_name = get_valid_str(row.get('Target Name'))
                        t_act = get_valid_str(row.get('Target Actions'))
                        t_gene = get_valid_str(row.get('Gene Name'))

                        target_entry = f"{t_act} of {t_name} (Gene: {t_gene})" if t_name else ""

                        if drug_name not in self.kb_dict:
                            self.kb_dict[drug_name] = {
                                "atc_info": atc or "Standard Class",
                                "description": desc or "Pharmacological profile available.",
                                "indication": ind or "Clinical compliance criteria applied.",
                                "mechanism": mech or "Mechanism detailed in baseline repository.",
                                "toxicity": tox or "Standard metabolic monitoring recommended.",
                                "targets": [target_entry] if target_entry else [],
                                "gene_names": [t_gene.upper()] if t_gene else []
                            }
                        else:
                            existing = self.kb_dict[drug_name]
                            if atc and existing["atc_info"] in ["Standard Class", ""]:
                                existing["atc_info"] = atc
                            if desc and existing["description"] in ["Pharmacological profile available.", ""]:
                                existing["description"] = desc
                            if ind and existing["indication"] in ["Clinical compliance criteria applied.", ""]:
                                existing["indication"] = ind
                            if mech and existing["mechanism"] in ["Mechanism detailed in baseline repository.", ""]:
                                existing["mechanism"] = mech
                            if tox and existing["toxicity"] in ["Standard metabolic monitoring recommended.", ""]:
                                existing["toxicity"] = tox

                            if target_entry and target_entry not in existing["targets"]:
                                existing["targets"].append(target_entry)
                            if t_gene and t_gene.upper() not in existing["gene_names"]:
                                existing["gene_names"].append(t_gene.upper())

                    print(
                        f"✅ Successfully reorganized [Multi-row and multi-target integrated knowledge base]: {CONFIG['KB_CSV']}，Total included after merging {len(self.kb_dict)} Characteristics of weight-removing drugs.")
                    break
                except Exception:
                    continue
        else:
            print(f"❌ Error: Core knowledge base file not found in current directory{CONFIG['KB_CSV']}！")

        self.session = self._setup_session()

    def _setup_session(self):
        session = requests.Session()
        session.mount('http://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
        return session

    def get_context(self, row):
        raw_mcc = str(row['MCC ']).strip()
        mcc_label = raw_mcc.replace('-', ' ').replace('_', ' ').upper()

        if len(mcc_label) == 4 and mcc_label.startswith("MCC"):
            mcc_label = f"MCC {mcc_label[-1]}"

        disease_keys = MCC_MAP.get(mcc_label, [])

        guideline_evidence = []
        for dk in disease_keys:
            path = os.path.join(CONFIG["TEMP_KB_DIR"], f"{dk}_kb.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    disease_data = json.load(f)
                    guideline_evidence.append(disease_data.get(dk, ""))

        protein_insights = []
        mcc_targets_set = set()
        for col in ['C1 (Therapeutic Efficacy Synergy)', 'C5 (Target Misdirection Penalty)']:
            raw = str(row[col])
            if raw.lower() != 'none':
                mcc_targets_set.update([t.split('(')[0].strip().upper() for t in raw.split(';')])

        for t in mcc_targets_set:
            info = self.prot_data.get(t, "")
            if info:
                clean_info = info.split('\n', 1)[-1].strip() if "Okay" in info[:20] else info
                protein_insights.append(f"Target [{t}]: {clean_info[:500]}")

        kb_evidence = []
        individual_drugs = [d.strip().lower() for d in str(row['Drugs']).split(';')]

        kbs_matched = 0
        for drug in individual_drugs:
            if drug in self.kb_dict:
                kbs_matched += 1
                info = self.kb_dict[drug]

                
                resonance_tags = [g for g in info["gene_names"] if g in mcc_targets_set]

                drug_genes_str = ", ".join(info["gene_names"]) if info["gene_names"] else "None recorded"
                mcc_genes_str = ", ".join(mcc_targets_set) if mcc_targets_set else "None recorded"

                if resonance_tags:
                    resonance_alert = f"✅ [TARGET RADAR MATCH]: Confirmed structural intersection. Drug targets ({', '.join(resonance_tags)}) directly map to the MCC pathology network targets."
                else:
                    resonance_alert = f"☑️ [TARGET RADAR SCANNED]: Cross-checked drug targets ({drug_genes_str}) against MCC targets ({mcc_genes_str}). No direct molecular overlap detected; operates via systemic mechanisms."

                all_targets_str = "; ".join(info['targets']) if info['targets'] else "Systemic pathway modulation."

                kb_evidence.append(
                    f"■ DRUG ENTITY: {drug.upper()}\n"
                    f"  - ATC Class: {info['atc_info']}\n"
                    f"  - General Description: {info['description'][:400]}\n"
                    f"  - Clinical Indication Profile: {info['indication'][:400]}\n"
                    f"  - Pharmacological Mechanism: {info['mechanism'][:600]}\n"
                    f"  - Documented Toxicity & Side Effects: {info['toxicity'][:600]}\n"
                    f"  - Molecular Target Profiling: {all_targets_str}\n"
                    f"  - {resonance_alert}"
                )
            else:
                kb_evidence.append(f"■ DRUG ENTITY: {drug.upper()}\n  - Status: Generic reference applied.")

        return "\n\n".join(guideline_evidence), "\n".join(protein_insights), "\n\n".join(kb_evidence), len(
            individual_drugs), kbs_matched

    def call_llm(self, prompt):
        payload = {
            "model": CONFIG["MODEL_NAME"],
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "system": (
                "You are an Advanced Clinical Pharmacologist and Medication Auditing Layer. "
                "Your task is to synthesize the provided empirical profiles (Evidence 1, 2, and 3) "
                "with standard clinical pharmacology principles to evaluate how effectively and safely "
                "the candidate drug regimen manages the patient's comprehensive metabolic comorbidity cluster (MCC)."
            ),
            "options": {
                "num_ctx": 30000,
                "temperature": 0.0
            }
        }
        try:
            res = self.session.post(CONFIG["OLLAMA_URL"], json=payload, timeout=500)
            return json.loads(res.json().get('response', '{}'))
        except Exception:
            return None

    def run_audit(self):
        if os.path.exists(CONFIG["OUTPUT_CSV"]):
            df = pd.read_csv(CONFIG["OUTPUT_CSV"])
        else:
            df = pd.read_csv(CONFIG["INPUT_CSV"])
            for col in ["Therapeutic_Coordination", "Contraindications_Risk", "Clinical_Guidance",
                        "Final_Pharmacist_Score"]:
                df[col] = ""

        mask = df['Number of drug combinations'].isin([2, 3, 4])
        targets = df[mask].head(CONFIG["EVAL_TOP_N"]).index

        print(f"👨‍⚕️ Standard Compliance Auditor starting deterministic review for {len(targets)} combinations...")

        counter = 0
        for idx in targets:
            row = df.loc[idx]
            if pd.notna(df.at[idx, 'Final_Pharmacist_Score']) and str(
                    df.at[idx, 'Final_Pharmacist_Score']).strip() != "":
                continue

            start_time = time.time()
            guide_ev, protein_ev, kb_ev, total_drugs, kbs_matched = self.get_context(row)
            safe_mcc = str(row['MCC ']).strip().replace('-', ' ').upper()

            prompt = f"""
            ### TASK: DETERMINISTIC REGIMEN COMPLIANCE & SAFETY AUDIT
            - Regimen to Verify: {row['Drugs']}
            - Clinical Cluster Profile: {row['MCC ']} (Target Diseases in this Cluster: {', '.join(MCC_MAP.get(safe_mcc, []))})

            ### EVIDENCE 1: PROTEIN SYSTEMIC CROSS-TALK TRAJECTORY (MCC PATHOLOGY)
            {protein_ev if protein_ev else "No direct protein anomalies recorded."}

            ### EVIDENCE 2: REGIONAL PRACTICE GUIDELINES & CONTRAINDICATION CRITERIA
            {guide_ev[:4500] if guide_ev else "Standard guideline background criteria applied."}

            ### EVIDENCE 3: DRUG-SPECIFIC REGULATORY PROFILES & TOXICITY ANNOTATIONS (With Target Resonance Radars)
            {kb_ev}

            ### EVALUATION PROTOCOL:
            1. Analyze coordination: Assess how the mechanisms of the evaluated drugs complement each other to address the multiple combined diseases within this specific MCC. (Note: Individual drugs may target different distinct diseases within the cluster; evaluate their multi-target therapeutic synergy as a unified combination). Factor in any [TARGET RADAR MATCH] alerts in Evidence 3 as highly robust structural synergy.
            2. Hard Toxicity Screening: Check if the documented toxicities or molecular target profiles of any drug in Evidence 3 negatively interact or create severe risk patterns for any of the active comorbidity domains in this MCC.
            3. Final Scoring Rule: Award an integer score from 1 to 10 based on the evidence. A score of 10 indicates perfect guideline-compliant synergy, verified target intervention, and absolute clinical safety across all cluster diseases. A score of 1 indicates critical toxic overlap, severe side effect contradiction, or absolute mechanism redundancy.

            ### COMPLIANT OUTPUT FORMAT SPECIFICATION (English JSON ONLY):
            {{
                "coordination": "Evidence-driven logical synthesis analyzing how the mechanisms of these drugs cross-talk or divide labor to treat the composite diseases of this MCC.",
                "risk": "Regimen-specific hazard warnings detailing recorded drug toxicities from Evidence 3 that might complicate the multi-morbid profile.",
                "guidance": "1-sentence strict directive per drug + explicit required lab monitoring markers mentioned or logically required based on evidence.",
                "score": "Final Score (1-10, an integer)."
            }}
            """

            result = self.call_llm(prompt)
            latency = time.time() - start_time

            success = result is not None and 'score' in result
            score = result.get('score', None) if success else None

            self.tracker.log_call(latency, success, total_drugs, kbs_matched, score)

            if result:
                df.at[idx, 'Therapeutic_Coordination'] = result.get('coordination', '')
                df.at[idx, 'Contraindications_Risk'] = result.get('risk', '')
                df.at[idx, 'Clinical_Guidance'] = result.get('guidance', '')
                df.at[idx, 'Final_Pharmacist_Score'] = score

                counter += 1
                print(f"✅ [{counter}] Rank {row['Rank']} Reviewed successfully at T=0.0. EES Score: {score}")

                if counter % CONFIG["SAVE_INTERVAL"] == 0:
                    df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding='utf-8-sig')
                    self.tracker.print_reviewer_metrics_panel()

        df.to_csv(CONFIG["OUTPUT_CSV"], index=False, encoding='utf-8-sig')
        print("Comprehensive deterministic drug audit completed!")
        self.tracker.print_reviewer_metrics_panel()


if __name__ == "__main__":
    agent = MORPH_Pharmacist_Agent()
    agent.run_audit()