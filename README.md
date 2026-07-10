              
                                           MORPH Framework
                  
                        ███╗   ███╗ ██████╗ ██████╗ ██████╗ ██╗  ██╗
                        ████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║  ██║
                        ██╔████╔██║██║   ██║██████╔╝██████╔╝███████║
                        ██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║
                        ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║
                        ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
                  
                             [AI] --> ( proteome reversal ) --> [ Score ]
                       
                        MORPH: AI-Driven Molecular Morphology & DPI Predictor
                        [■] TASK: Drug-Protein Interaction [■] VER: 1.1.0
                        [■] AUTH: Shixuan.Z & ZhenQiu.L    [■] SYS: Mac M or Win 10

    
# 🧬 MCC: LLM-Based Pharmacological Auditing Agent 
### For Metabolic Comorbidity Clusters
### MORPH model assistance
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Model-Gemma--3--12B-orange.svg" alt="Model">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>
------------------------------------------------------------------------------------

It is an advanced, deterministic Large Language Model (LLM) agent designed for the automated pharmacological auditing of complex multi-drug regimens. Specifically tailored for Metabolic Comorbidity Clusters (MCCs), the MORPH reasoning engine bridges the gap between proteomic prioritization and real-world clinical feasibility.

------------------------------------------------------------------------------------
## 🌟Key Features
* **Zero-Drift Deterministic Inference:** Completely eliminates the probabilistic illusion of large models by running with strictly controlled core parameters (temperature coefficient `Temperature T=0.0`). Achieving **100.0% absolute zero drift rate (SD = 0.0000)** in independent iterations, serving as an extremely robust and verifiable clinical compliance filter.
* **Multi-Dimensional Knowledge Base:** Seamlessly integrates real-world clinical and multi-omics data, including:
* **Clinical Guidelines:** Extracted and structured international clinical medical guideline texts covering 6 basic metabolic diseases (type 2 diabetes, hyperlipidemia, hypertension, fatty liver, hyperuricemia and obesity).
* ** Authoritative drug data: ** Integrate detailed descriptions of **343** target drugs in DrugBank (latest version in January 2026), covering indications (Indications), mechanism of action (MoA), toxicity (Toxicity) and target effects (Target Actions).
* **Proteomics:** Incorporates protein pathophysiological topology and literature-derived functional descriptions of marker proteins.
* **Core Disease Coverage Protocol:** The inference engine strictly enforces the **80% core disease coverage threshold**. If the current multi-drug regimen fails to meet the basic coverage baseline of the comorbidity cluster, the agent will forcibly trigger mandatory drug replacement or optimized addition recommendations.
* **Explainable Evaluation Score (EES):** Outputs a highly interpretable **1–10** comprehensive metric (EES), which conducts quantitative evaluation from the three dimensions of drug interactions (DDIs), mechanism redundancy (Mechanistic Redundancy), and therapeutic synergy (Therapeutic Synergy), and is accompanied by detailed clinical evidence and pharmacological reasoning processes.

------------------------------------------------------------------------------------

## 📊 Clinical Validation & Performance

The clinical reliability of the EES has been rigorously benchmarked against human expertise through an independent, double-blinded cross-validation protocol involving two senior clinical pharmacists.

| Evaluation Domain | Statistical Metric | Result | P-value / 95% CI |
| :--- | :--- | :---: | :---: |
| **Inter-rater Reliability** | Intraclass Correlation (ICC3,1) | **0.745** | 95% CI: 0.49 - 0.87 |
| | Quadratic Weighted Kappa | **0.731** | $p = 8.42 \times 10^{-8}$ |
| **Human-AI Alignment** | Pearson Correlation (r) | **0.727** | $p = 7.28 \times 10^{-9}$ |
| | Spearman Correlation ($\rho$) | **0.629** | $p = 2.18 \times 10^{-6}$ |
| **Error Metrics** | Mean Absolute Error (MAE) | **1.085** Points | — |
| | Root Mean Square Error (RMSE) | **1.455** Points | — |

> 💡 **Conclusion:** These metrics demonstrate substantial objective agreement between the deterministic LLM outputs and human expert consensus. This proves the LLM's potential as a highly reliable tool for screening out regimens that are theoretically favorable on a proteomic scale but lack real-world clinical safety.
---

## 📂 Repository Structure

```text
MCC-LLM-based-auditing-module/
├── data/
│   ├── DrugBank_Knowledge_Base.csv                       # Features for 343 drugs (Indication, MoA, Toxicity)
│   ├── Prot_KB.json                                      # MCC definitions and basal disease mapping
│   ├── guide/                                            # Guide files and knowledge base json
│   └── Rawdata/                                          # MCC related basic files
├── src/
│   ├── Evaluate_MCC_Combinations.py                      # Agent running file (please enter API key here)
│   └── Test_LLM_Temperature/                             # LLM Temperature Test scripts
├── results/
│   ├── Table S19——Pharmacist_Audit.csv                   # Inference results (Table S19)
│   └── Supplementary_Table_S26_Repeatability_Benchmark.csv # 25-regimen stability benchmark (Table S31)
├── text/
│   └── Table S19.csv                                     # Test file
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------------------
🚀 Quick Start
------------------------------------------------------------------------------------
1. Environment Setup
---------------------------
```text
git clone https://github.com/ShixuanFudan/MCC-LLM-based-auditing-module.git
cd MCC-LLM-based-auditing-module
conda create -n morph_env python=3.10
conda activate morph_env
pip install -r requirements.txt
```
---------------------------
2. Configure the LLM Agent
Ensure you have the appropriate access to the Gemma-3 12B model weights or API. Update your configuration in src/agent_core.py to set T=0.0 for deterministic outputs.
---------------------------
3. Run a Sample Audit
```text
python src/Evaluate_MCC_Combinations.py
```

------------------------------------------------------------------------------------
🤝 Acknowledgments
------------------------------------------------------------------------------------

We extend our deepest gratitude to the clinical pharmacists and collaborators who made the stringent double-blind validation possible:

Shixuan.Z  (Algorithm Implementation)
ZhenQiu.L (Algorithm Implementation)
Jingru.G (Clinical Pharmacist, Independent Blinded Rater)
Shun.S (Clinical Pharmacist, Independent Blinded Rater)

##
If you have any questions, suggestions or academic cooperation intentions, please contact us via email:
📧 Email: sxzhang21@m.fudan.edu.cn



