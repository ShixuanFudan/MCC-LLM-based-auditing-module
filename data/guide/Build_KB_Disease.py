import json
import os
import requests
import pdfplumber
import time
import argparse

# =================================================================
#                         1. 配置区
# =================================================================
CONFIG = {
    "MODEL_NAME": "gemma3:12b",
    "OLLAMA_URL": "http://localhost:11434/api/generate",
    "TEMP_DIR": "temp_kbs",
    "MASTER_KB": "mcc_clinical_pharmacist_kb.json",
    # 加入了中文的常见药理章节标题，进一步提高中文 PDF 的定位准确率
    "PHARMA_KEYWORDS": [
        "Pharmacologic", "Drug Therapy", "Treatment Algorithm", "Medication",
        "Combination Therapy", "Dosing", "Therapeutics", "Medical Management",
        "药物治疗", "用药", "药理", "治疗方案", "联合治疗", "降压药物", "降糖药物"
    ]
}


class SmartGuidelineExtractor:
    def __init__(self, disease_key):
        self.disease_key = disease_key
        self.temp_file = os.path.join(CONFIG["TEMP_DIR"], f"{disease_key}_kb.json")
        if not os.path.exists(CONFIG["TEMP_DIR"]):
            os.makedirs(CONFIG["TEMP_DIR"])

    def call_ollama(self, prompt, context):
        payload = {
            "model": CONFIG["MODEL_NAME"],
            "prompt": f"### CONTEXT FROM GUIDELINE:\n{context}\n\n### TASK:\n{prompt}",
            "stream": False,
            "system": "You are a Senior Clinical Pharmacist and a bilingual medical expert (Chinese/English). Your output must be structural English summaries.",
            "options": {"num_ctx": 24000, "temperature": 0.1}
        }
        try:
            res = requests.post(CONFIG["OLLAMA_URL"], json=payload, timeout=400)
            return res.json().get("response", "")
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def find_core_pages(self, pdf):
        """安全扫描 PDF，寻找包含药理核心关键词的页码 (兼容中英文)"""
        core_pages = []
        total_pages = len(pdf.pages)
        print(f"  🔍 正在扫描关键词定位核心章节 (共 {total_pages} 页)...")

        for i, page in enumerate(pdf.pages):
            extracted = page.extract_text()
            header_text = extracted[:400] if extracted else ""

            if any(key.lower() in header_text.lower() for key in CONFIG["PHARMA_KEYWORDS"]):
                core_pages.append(i)

        if not core_pages:
            print("  ⚠️ 未找到明确的药理章节关键词，将默认提取前 25 页...")
            end_fallback = min(total_pages, 25)
            return range(0, end_fallback)

        start = max(0, core_pages[0])
        end = min(total_pages, start + 25)
        print(f"  📍 自动定位核心药理章节：第 {start} 至 {end} 页")
        return range(start, end)

    def extract(self, folder_path):
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        all_summaries = []

        for fname in pdf_files:
            file_path = os.path.join(folder_path, fname)
            print(f"📖 处理指南文件: {fname}")

            with pdfplumber.open(file_path) as pdf:
                target_pages = self.find_core_pages(pdf)
                text_content = ""
                for p_idx in target_pages:
                    if p_idx < len(pdf.pages):
                        page_text = pdf.pages[p_idx].extract_text()
                        if page_text:
                            text_content += page_text + "\n"

            # 🌟 核心修改：明确告知 LLM 存在中文，并强制要求英文药理学提炼
            prompt = """
                        You are a Clinical Pharmacist. The provided guideline text may be in ENGLISH or CHINESE.
                        Regardless of the source language, extract PRECISE pharmacological rules into structured ENGLISH ONLY:

                        1. **Drug Priority**: First-line vs Second-line logic.
                        2. **Synergy**: Which drugs are recommended to be combined? 
                        3. **Redundancy/Avoidance**: Which drugs MUST NOT be used together?
                        4. **Safety Thresholds**: Specific eGFR, CrCl, or LFT cut-offs for dosing adjustment.
                        5. **Monitoring**: Key biomarkers to check during titration.

                        CRITICAL CONSTRAINTS:
                        - DO NOT include conversational filler (e.g., "Here is the summary", "Okay").
                        - DO NOT include disclaimers or notes about the limitations of the text.
                        - If you encounter a table, EXTRACT THE ACTUAL VALUES (e.g., specific lab tests, specific side effects). DO NOT just say "the table provides an overview".
                        - Output MUST be 100% hard clinical facts.
                        """
            print(f"  🧠 交给 LLM 进行跨语言提炼药理学逻辑...")
            summary = self.call_ollama(prompt, text_content[:20000])
            all_summaries.append(f"### [Source File: {fname}]\n{summary}")

        with open(self.temp_file, 'w', encoding='utf-8') as f:
            json.dump({self.disease_key: "\n\n".join(all_summaries)}, f, ensure_ascii=False, indent=4)
        print(f"✅ 已生成 {self.disease_key} 的专业知识模块: {self.temp_file}")


# =================================================================
#                         2. 运行与合并
# =================================================================
def merge_logic():
    print("\n🔗 开始合并所有疾病知识库...")
    if not os.path.exists(CONFIG["MASTER_KB"]):
        master_data = {
            "metadata": {"brand": "MORPH", "created_at": time.strftime("%Y-%m-%d")},
            "guidelines": {},
            "proteins": {},
            "drug_atc": {}
        }
    else:
        with open(CONFIG["MASTER_KB"], 'r', encoding='utf-8') as f:
            master_data = json.load(f)

    temp_files = [f for f in os.listdir(CONFIG["TEMP_DIR"]) if f.endswith('_kb.json')]
    for tf in temp_files:
        with open(os.path.join(CONFIG["TEMP_DIR"], tf), 'r', encoding='utf-8') as f:
            disease_content = json.load(f)
            master_data["guidelines"].update(disease_content)

    if "HLT" in master_data.get("guidelines", {}):
        del master_data["guidelines"]["HLT"]

    with open(CONFIG["MASTER_KB"], 'w', encoding='utf-8') as f:
        json.dump(master_data, f, ensure_ascii=False, indent=4)
    print(f"✨ 合并完成！最终知识库已保存至: {CONFIG['MASTER_KB']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_logic()
    elif args.disease and args.folder:
        SmartGuidelineExtractor(args.disease).extract(args.folder)
    else:
        print(
            "💡 使用说明: \n 提取: python Build_KB_Disease.py --disease HTN --folder ./guide/HTN \n 合并: python Build_KB_Disease.py --merge")