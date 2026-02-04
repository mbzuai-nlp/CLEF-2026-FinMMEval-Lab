# FinMMEval Lab @ CLEF 2026

Multilingual and multimodal evaluation of financial AI systems. The lab spans three complementary tasks and emphasizes evidence-grounded outputs with calibrated confidence.

## Tasks
- **Task 1 – Financial Exam Q&A:** Evaluates conceptual understanding and domain reasoning using multilingual, professional exam-style financial questions (e.g., CFA, CPA, EFPA, BBF). Performance is measured by accuracy.
- **Task 2 – Multilingual Financial Q&A:** Tests analytical financial reasoning using multilingual and multimodal information sources (e.g., SEC filings plus cross-lingual news). Models generate concise, evidence-grounded answers evaluated with ROUGE and factuality metrics.
- **Task 3 – Financial Decision Making:** Assesses reasoning-to-action by generating Buy/Hold/Sell decisions and short rationales from textual and numerical market contexts (BTC, TSLA). Evaluated via profitability, stability, and risk metrics (e.g., Sharpe Ratio, Cumulative Return).

## What to Submit
- Per-task prediction JSONL files with confidence scores.
- Evidence trace or rationale aligned to task definitions.
- System card (model design, data usage, risks) and reproducibility notes (seed, versions, hardware).

## How to Participate
- Register via the official CLEF form (choose FinMMEval tasks): https://clef-labs-registration.dipintra.it/registrationForm.php#registrationFrom
- Task 3 endpoint submission (Agent Market Arena Google Form): https://huggingface.co/spaces/TheFinAI/Agent-Market-Arena
- Task 3 submission deadline: 28 April 2026
- Full call, visuals, and timeline: https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/

## Training Data (Released)
- Download the training collection on Hugging Face (released 2025-12-15): https://huggingface.co/collections/MBZUAI/finmmeval-lab-clef2026
- Contains training splits for all three tasks (exam Q&A, multilingual Q&A, and trading decision making). See each dataset card in the collection for licenses and format details.
- Task 3 historical data for backtesting, validation, and training: TheFinAI/CLEF_Task3_Trading

## Awards
- 🏆 Best Paper Award: USD 500
- 🥈 Outstanding Paper Award ×3: USD 300 each
- 🌱 Merit / Encouragement Award ×2: USD 200 each

## Contact
- Email: zhuohan.xie@mbzuai.ac.ae
- Discord: https://discord.gg/xR23AbBy

## Organizers
- Zhuohan Xie (MBZUAI, UAE)
- Rania Elbadry (MBZUAI, UAE)
- Fan Zhang (The University of Tokyo, Japan)
- Georgi Georgiev (Sofia University "St. Kliment Ohridski", Bulgaria)
- Xueqing Peng (The Fin AI, USA)
- Lingfei Qian (The Fin AI, USA)
- Jimin Huang (The Fin AI, USA)
- Dimitar Dimitrov (Sofia University "St. Kliment Ohridski", Bulgaria)
- Vanshikaa Jani (University of Arizona, USA)
- Yuyang Dai (INSAIT, Bulgaria)
- Jiahui Geng (Linköping University, Sweden)
- Yankai Chen (McGill University, Canada & MBZUAI, UAE)
- Yuan Ye (McGill University, Canada & Mila - Quebec AI Institute, Canada)
- Haolun Wu (McGill University, Canada & Mila - Quebec AI Institute, Canada)
- Yuxia Wang (INSAIT, Bulgaria)
- Ivan Koychev (Sofia University "St. Kliment Ohridski", Bulgaria)
- Veselin Stoyanov (MBZUAI, UAE)
- Mingzi Song (Nikkei Financial Technology Research Institute, Inc., Japan)
- Yu Chen (The University of Tokyo, Japan)
- Steve Liu (McGill University, Canada & MBZUAI, UAE)
- Preslav Nakov (MBZUAI, UAE)

Questions? Reach out to the organizers listed above or via the website.
