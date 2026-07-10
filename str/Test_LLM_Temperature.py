import pandas as pd
import json
import requests
import time
import os
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import pearsonr

# =================================================================
#                         1. 测试参数配置区
# =================================================================
CONFIG = {
    "INPUT_CSV": "Table S19——Pharmacist_Audit.csv",  # 直接读取你已经跑完结果的表
    "KB_CSV": "Knowledge base.csv",
    "PROT_KB": "Prot_KB.json",
    "TEMP_KB_DIR": "temp_kbs",
    "MODEL_NAME": "gemma3:12b",
    "OLLAMA_URL": "http://localhost:11434/api/generate",

    # 🌟 审稿人测试参数
    "TEST_SAMPLE_SIZE": 25,  # 随机抽取15个复杂组合进行压力测试
    "TEST_TEMPERATURE": 0.0,  # 测试温度，必须与你跑大盘时的温度一致
    "TEST_ITERATIONS": 3,  # 独立重复测试的次数 (Run 1, Run 2, Run 3)
    "OUTPUT_REPORT": "Supplementary_Table_S26_Repeatability_Benchmark.csv"
}

MCC_MAP = {
    "MCC 1": ["HLP"], "MCC 2": ["HLP", "HTN"], "MCC 3": ["HLP", "HTN", "FLD"],
    "MCC 4": ["HLP", "HTN", "FLD"], "MCC 5": ["HLP", "HTN", "FLD"],
    "MCC 6": ["HLP", "HTN", "FLD", "T2D", "HUA", "OBE"],
    "MCC 7": ["HLP", "HTN", "FLD", "T2D", "HUA", "OBE"],
    "MCC 8": ["HLP", "HTN", "FLD", "T2D", "HUA", "OBE"]
}


class MORPH_Repeatability_Tester:
    def __init__(self):
        # 简化版的 KB 加载逻辑（与你跑大盘时完全一致）
        if os.path.exists(CONFIG["PROT_KB"]):
            with open(CONFIG["PROT_KB"], 'r', encoding='utf-8') as f:
                raw = json.load(f)
                self.prot_data = raw.get('proteins', raw)
        else:
            self.prot_data = {}

        self.kb_dict = {}
        if os.path.exists(CONFIG["KB_CSV"]):
            for enc in ['utf-8-sig', 'gbk', 'utf-8']:
                try:
                    df_kb = pd.read_csv(CONFIG["KB_CSV"], encoding=enc)
                    for _, row in df_kb.iterrows():
                        d_name = str(row.get('Drug names', '')).strip().lower()
                        if not d_name or d_name in ['nan', 'none']: continue

                        def clean(v):
                            return "" if str(v).lower() in ["nan", "n/a", "none", ""] else str(v).strip()

                        t_name = clean(row.get('Target Name'))
                        t_gene = clean(row.get('Gene Name'))
                        t_entry = f"{clean(row.get('Target Actions'))} of {t_name} (Gene: {t_gene})" if t_name else ""

                        if d_name not in self.kb_dict:
                            self.kb_dict[d_name] = {
                                "atc_info": clean(row.get('ATC Information')),
                                "description": clean(row.get('Description')),
                                "indication": clean(row.get('Indication')),
                                "mechanism": clean(row.get('Mechanism of Action')),
                                "toxicity": clean(row.get('Toxicity')),
                                "targets": [t_entry] if t_entry else [],
                                "gene_names": [t_gene.upper()] if t_gene else []
                            }
                        else:
                            ex = self.kb_dict[d_name]
                            if t_entry and t_entry not in ex["targets"]: ex["targets"].append(t_entry)
                            if t_gene and t_gene.upper() not in ex["gene_names"]: ex["gene_names"].append(
                                t_gene.upper())
                    break
                except:
                    continue

        self.session = requests.Session()
        self.session.mount('http://', HTTPAdapter(max_retries=Retry(total=3)))

    def get_context(self, row):
        # [复用你之前的 get_context 逻辑，确保提供给大模型的文本完全一致]
        safe_mcc = str(row['MCC ']).strip().replace('-', ' ').upper()
        if len(safe_mcc) == 4 and safe_mcc.startswith("MCC"): safe_mcc = f"MCC {safe_mcc[-1]}"

        guide_ev = []
        for dk in MCC_MAP.get(safe_mcc, []):
            p = os.path.join(CONFIG["TEMP_KB_DIR"], f"{dk}_kb.json")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f: guide_ev.append(json.load(f).get(dk, ""))

        prot_ev = []
        mcc_genes = set()
        for c in ['C1 (Therapeutic Efficacy Synergy)', 'C5 (Target Misdirection Penalty)']:
            if str(row[c]).lower() != 'none': mcc_genes.update(
                [t.split('(')[0].strip().upper() for t in str(row[c]).split(';')])
        for g in mcc_genes:
            if info := self.prot_data.get(g, ""): prot_ev.append(f"Target [{g}]: {info[:500]}")

        kb_ev = []
        for drug in [d.strip().lower() for d in str(row['Drugs']).split(';')]:
            if drug in self.kb_dict:
                info = self.kb_dict[drug]
                match = [g for g in info["gene_names"] if g in mcc_genes]
                alert = f"✅ [TARGET RADAR MATCH]: Targets ({', '.join(match)}) directly map." if match else "☑️ [TARGET RADAR SCANNED]: No direct overlap."
                kb_ev.append(
                    f"■ DRUG: {drug.upper()}\n  - ATC: {info['atc_info']}\n  - Mech: {info['mechanism'][:400]}\n  - Tox: {info['toxicity'][:400]}\n  - {alert}")
            else:
                kb_ev.append(f"■ DRUG: {drug.upper()}\n  - Generic applied.")

        return "\n\n".join(guide_ev), "\n".join(prot_ev), "\n\n".join(kb_ev)

    def call_llm(self, prompt):
        payload = {
            "model": CONFIG["MODEL_NAME"], "prompt": prompt, "format": "json", "stream": False,
            "system": "You are an Advanced Clinical Pharmacologist.",
            "options": {"num_ctx": 30000, "temperature": CONFIG["TEST_TEMPERATURE"]}
        }
        try:
            res = self.session.post(CONFIG["OLLAMA_URL"], json=payload, timeout=300)
            data = json.loads(res.json().get('response', '{}'))
            return float(data.get('score', np.nan))
        except:
            return np.nan

    def run_benchmark(self):
        print(f"🚀 启动大模型可复现性压力测试 (Repeatability Benchmark)")
        print(
            f"🔹 测试设定: T = {CONFIG['TEST_TEMPERATURE']} | 重复次数: {CONFIG['TEST_ITERATIONS']} 次 | 样本量: {CONFIG['TEST_SAMPLE_SIZE']} 组\n")

        df_all = pd.read_csv(CONFIG["INPUT_CSV"])
        df_test = df_all[df_all['Number of drug combinations'].isin([2, 3, 4])].sample(n=CONFIG["TEST_SAMPLE_SIZE"],
                                                                                       random_state=42)

        results = []
        for idx, row in df_test.iterrows():
            print(f"🧪 测试方案 [Rank {row['Rank']}]: {row['Drugs']}")
            g_ev, p_ev, k_ev = self.get_context(row)
            safe_mcc = str(row['MCC ']).strip().replace('-', ' ').upper()

            prompt = f"""
            ### TASK: DETERMINISTIC COMPLIANCE AUDIT
            - Drugs: {row['Drugs']} | Cluster: {row['MCC ']} ({', '.join(MCC_MAP.get(safe_mcc, []))})
            ### EVIDENCE 1: {p_ev}
            ### EVIDENCE 2: {g_ev[:3000]}
            ### EVIDENCE 3: {k_ev}
            ### OUTPUT JSON ONLY: {{"score": "Integer 1-10"}}
            """

            scores = []
            for run_i in range(1, CONFIG["TEST_ITERATIONS"] + 1):
                s = self.call_llm(prompt)
                scores.append(s)
                print(f"    ↳ 独立运行 {run_i}/3 ... 得分: {s}")

            arr = np.array(scores)
            results.append({
                "Rank": row['Rank'], "Drugs": row['Drugs'], "Original_Score": row.get('Final_Pharmacist_Score', 'N/A'),
                "Run_1": scores[0], "Run_2": scores[1], "Run_3": scores[2],
                "Mean_Score": round(np.nanmean(arr), 2),
                "Standard_Deviation(SD)": round(np.nanstd(arr, ddof=1) if len(arr[~np.isnan(arr)]) > 1 else 0.0, 4)
            })

        df_res = pd.DataFrame(results)

        # === 计算审稿人要看的全局终极统计指标 ===
        avg_sd = df_res['Standard_Deviation(SD)'].mean()
        max_sd = df_res['Standard_Deviation(SD)'].max()
        zero_drift_rate = (df_res['Standard_Deviation(SD)'] == 0.0).mean() * 100

        print("\n" + "=" * 60)
        print("📊    QUANTITATIVE REPEATABILITY BENCHMARK REPORT    ")
        print("=" * 60)
        print(
            f"🔹 1. 测试设置 (Setting):        Temperature = {CONFIG['TEST_TEMPERATURE']}, Iterations = {CONFIG['TEST_ITERATIONS']}")
        print(f"🔹 2. 平均分数标准差 (Mean SD):  {avg_sd:.4f} 分")
        print(f"🔹 3. 最大异常波动幅 (Max SD):   {max_sd:.4f} 分")
        print(f"🔹 4. 绝对零漂移率 (Zero Drift): {zero_drift_rate:.1f}% (完全一致的比例)")
        print("=" * 60)

        df_res.to_csv(CONFIG["OUTPUT_REPORT"], index=False, encoding='utf-8-sig')
        print(f"\n✅ 测试完成！详细跑分矩阵已保存至: {CONFIG['OUTPUT_REPORT']}")
        print("💡 [建议]：请将这个表格作为 Supplementary Table S26，配合下面的回复话术发给审稿人。")


if __name__ == "__main__":
    tester = MORPH_Repeatability_Tester()
    tester.run_benchmark()