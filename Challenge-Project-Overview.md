# Compliance Copilot: Auditing Software Requirements Against Engineering Standards

**Company / Org:** Break Through Tech AI Studio  
**Challenge Advisor:** Beth Parnell elizabeth.parnell@breakthroughtech.org  
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About NASA & Regulated Engineering Standards
In safety-critical software engineering—such as NASA space missions, medical device development, and financial infrastructure—every single software requirement must undergo a rigorous, clause-by-clause manual audit against strict compliance standards (e.g., NASA NPR 7150.2) before code can be written. 

Currently, this compliance review relies on a small group of senior domain experts, creating a massive operational bottleneck and gatekeeping step. Automating this process transforms weeks of tedious manual review into rapid, structured first-pass reports that any engineer on the team can inspect and verify.

---

## 🎯 The Challenge

### Project Summary
The team will build an AI-powered **Compliance Copilot** that reads software requirements, retrieves relevant clauses from an engineering compliance standard, evaluates whether each requirement satisfies the standard, and generates a structured audit report. The project follows a clear two-stage architectural evolution:
1. **Non-Agentic Baseline:** A deterministic Retrieval-Augmented Generation (RAG) pipeline that queries relevant standard clauses and judges requirement compliance.
2. **Tool-Calling LLM Agent:** A dynamic agent equipped with tools (`retrieve_clause`, `check_requirement`, `log_gap`) that orchestrates the audit workflow and compiles final JSON and Markdown gap reports.

### Success Criteria
- **Day-One Ground-Truth Benchmark:** A human-labeled benchmark of requirement/clause pairs with explicit verdicts (`Meets`, `Partial`, `Gap`).
- **Grounded Verification:** Zero ungrounded or hallucinated verdicts; 100% of generated verdicts must cite the specific standard clause ID.
- **Quantitative Performance:** High Precision and Recall in identifying seeded non-compliance gaps compared against the ground-truth benchmark.

### Project Milestones
| Month | Milestone | Key Activities |
| :--- | :--- | :--- |
| **September** | Data Parsing & Day-One Benchmark Setup | Parse NASA NPR 7150.2 into structured clause chunks. Create a labeled ground-truth benchmark (30–50 requirement/clause pairs) with seeded compliance gaps and draft an accompanying Data Card. |
| **October** | Non-Agentic RAG Baseline & Schema Enforcement | Build an in-memory vector index (FAISS/Chroma). Implement a deterministic RAG retrieval pipeline using Pydantic JSON schemas to force structured citations and verdicts. Evaluate initial precision and recall. |
| **November** | Tool-Calling Agent & Automated Reporting | Wrap the pipeline into an LLM agent with dedicated tools (`retrieve_clause`, `check_requirement`, `log_gap`). Add API rate-limit caching and batching. Generate full JSON and Markdown gap reports. |
| **December** | Evaluation, Error Analysis & Stretch Horizons | Conduct quantitative benchmark evaluation comparing Baseline vs. Agent performance. Perform qualitative error analysis. Executing stretch goals (LangGraph multi-agent refactoring or cross-standard generalization). |

---

## 📊 Dataset

**Name and Source:** NASA Software Engineering Requirements NPR 7150.2D (https://nodis3.gsfc.nasa.gov/npg_img/N_PR_7150_002D_/N_PR_7150_002D_.pdf) + Synthetic Software Requirements Specification Dataset  
**Format:** Plain Text, JSON, CSV  
**Size:** < 1 GB (100% Google Colab Free Tier Compliant)  
**Location:** Public NASA NODIS Library & Student-Generated Evaluation Repository  

### Key Details
- **Compliance Standard (Answer Key):** NASA NPR 7150.2 document containing numbered, structured software engineering requirements.
- **Requirements to Audit:** Student-created software requirements set, intentionally containing compliant items and seeded non-compliance gaps (e.g., missing security specifications, vague testing criteria).
- **Data Card:** A documentation artifact explaining dataset curation, gap distribution, and labeling criteria.

---

## 🛠️ Suggested Approach

**ML Problem Type:** Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), LLM Tool-Calling Agents, Structured Output Enforcement.  
**Recommended Libraries:**
- `python` (Core language)
- `pandas` (Data manipulation and evaluation reporting)
- `pydantic` (JSON schema and output type validation)
- `langchain` / `faiss-cpu` / `chromadb` (Vector store, chunking, and retrieval)
- `langgraph` (Optional December stretch goal for multi-agent state graph)

**Evaluation Metrics:**
- **Precision:** Percentage of flagged compliance gaps that are true gaps.
- **Recall:** Percentage of actual compliance gaps correctly flagged by the system.
- **Citation Groundedness:** Proportion of model verdicts that correctly cite valid clause numbers.

---

## 📚 Resources to Get Started

**Background Reading:**
- NASA Procedural Requirements: *NPR 7150.2 Software Engineering Requirements*.
- Industry overview of AI applications in Governance, Risk, and Compliance (GRC).

**Technical Tutorials:**
- LangChain Documentation: *Retrieval-Augmented Generation (RAG) & Vector Stores*.
- LangChain / OpenAI / Gemini Guides: *Tool Calling and Function Output Structuring with Pydantic*.
- Grounding & Prompt Engineering: *Forcing Citations and Reducing Hallucination in Audit Reports*.

---

## 🤝 How We'll Work Together

**Check-ins:** Weekly technical lab sections and biweekly Challenge Advisor check-ins.  
**Communication:** BTT Project Discord Channel & GitHub Issue Tracker.  
**Response Time:** Within 24–48 hours for non-urgent technical questions.  

**Recommended Environment:**
- **Development Environment:** Google Colab (Free Tier CPU/T4 GPU).
- **Code Repository:** GitHub (Public project repository with modular Python scripts).
- **Deliverables:** Colab Prototype Notebook, Ground-truth Benchmark CSV, Data Card, JSON/Markdown Audit Report, and Reproducibility README.

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
