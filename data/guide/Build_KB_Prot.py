import pandas as pd
import json
import os
import requests
import pdfplumber
import time


class MORPH_KnowledgeBuilder_EN:
    def __init__(self, model_name="gemma3:12b"):
        self.brand = "MORPH"
        self.version = "1.2.0"
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.kb_file = "Prot_KB.json"

        print(f"🚀 {self.brand} Knowledge Base Builder Starting (Model: {self.model_name})")
        self._check_ollama_status()
        self._load_csv_resources()

    def _check_ollama_status(self):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama connection active. Ready to build KB.")
            else:
                print("❌ Error: Ollama service issue.")
        except Exception:
            print("❌ Error: Cannot connect to Ollama. Ensure 'ollama serve' is running.")

    def _load_csv_resources(self):
        """Load TableS11 and ATC_Annote with multi-encoding support"""
        resource_files = {"df_s11": "TableS9.csv", "df_atc": "Knowledge base.csv"}
        encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb18030']
        for attr, filename in resource_files.items():
            success = False
            if not os.path.exists(filename):
                print(f"⚠️ Missing file: {filename}")
                setattr(self, attr, pd.DataFrame())
                continue
            for enc in encodings:
                try:
                    setattr(self, attr, pd.read_csv(filename, encoding=enc))
                    success = True
                    break
                except:
                    continue
            if success: print(f"✅ Loaded {filename}")

    def call_ollama(self, prompt, context="", system_prompt=""):
        # Limit context to avoid timeout
        safe_context = context[:8000] if context else ""

        payload = {
            "model": self.model_name,
            "prompt": f"[Background Evidence]:\n{safe_context}\n\n[Instruction]:\n{prompt}",
            "stream": False,
            "system": system_prompt or "You are a world-class clinical pharmacologist. Provide evidence-based analysis in English ONLY."
        }
        try:
            # High timeout for 12B model reasoning
            response = requests.post(self.ollama_url, json=payload, timeout=180)
            return response.json().get("response", "")
        except Exception as e:
            return f"LLM Failure: {str(e)}"

    def process_guidelines(self, folder="guide"):
        """Extract guidelines in English"""
        guide_kb = {}
        if not os.path.exists(folder):
            print(f"⚠️ Folder '{folder}' not found.")
            return guide_kb

        pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
        for fname in pdf_files:
            key = fname.split(".")[0]
            print(f"📖 Analyzing Guideline: {fname}...")
            full_path = os.path.join(folder, fname)

            text_content = ""
            with pdfplumber.open(full_path) as pdf:
                for page in pdf.pages[:8]:  # Core treatment logic usually in first few pages
                    extracted = page.extract_text()
                    if extracted: text_content += extracted + "\n"

            prompt = """
            As a senior clinical pharmacist, extract and structure the following from this guideline in English:
            1. Treatment Pathways: Define first-line, second-line, and combination therapies.
            2. Key Medication Logic: Initial dosing and titration principles.
            3. Contraindications: Distinguish absolute and relative contraindications.
            4. Organ Function Adjustments: Specific dosing for renal (eGFR) and hepatic impairment.
            5. Monitoring: Key biochemical indicators (e.g., Creatinine, Electrolytes, Uric acid).
            Output must be a structured summary in English.
            """
            analysis = self.call_ollama(prompt, context=text_content)
            guide_kb[key] = analysis
            print(f"✅ Finished {key}")

        return guide_kb

    def process_protein_knowledge(self):
        """Convert Table S11 into clinical pathological insights in English"""
        protein_kb = {}
        if self.df_s11.empty: return protein_kb
        print("🧪 Processing Table S11 protein omics data...")

        for _, row in self.df_s11.iterrows():
            protein = str(row.get('Protein', 'Unknown'))
            impact = str(row.get('Impact', 'No data available'))

            prompt = f"""
            Analyze the pathophysiological role of protein {protein} in the metabolic comorbidity network.
            Known functional description: {impact}
            Please derive:
            1. Systemic pathological consequences resulting from the dysregulation of this protein.
            2. Expected clinical benefits of pharmacological intervention (agonism or antagonism) targeting this protein.
            Response must be in English.
            """
            insight = self.call_ollama(prompt,
                                       system_prompt="You are a bioinformatics and translational medicine expert.")
            protein_kb[protein] = insight
            print(f"🧬 Processed Protein: {protein}")

        return protein_kb

    def build_full_kb(self):
        """Integrate all knowledge into an English JSON KB"""
        start_time = time.time()

        full_kb = {
            "metadata": {
                "brand": self.brand,
                "version": self.version,
                "language": "English",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "guidelines": self.process_guidelines(),
            "proteins": self.process_protein_knowledge(),
            "drug_atc": self.df_atc.set_index('Drug names')[
                'ATC Information'].to_dict() if not self.df_atc.empty else {}
        }

        with open(self.kb_file, 'w', encoding='utf-8') as f:
            json.dump(full_kb, f, ensure_ascii=False, indent=4)

        duration = (time.time() - start_time) / 60
        print(f"\n🎉 KB Construction Successful! Time taken: {duration:.2f} mins")
        print(f"📂 Saved to: {self.kb_file}")


if __name__ == "__main__":
    builder = MORPH_KnowledgeBuilder_EN()
    builder.build_full_kb()