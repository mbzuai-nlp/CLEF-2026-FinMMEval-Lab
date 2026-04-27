# FinMMEval Lab @ CLEF 2026

Multilingual and multimodal evaluation of financial AI systems. The lab spans three complementary tasks and emphasizes evidence-grounded, task-appropriate outputs.

## Tasks
- **Task 1 – Financial Exam Q&A:** Evaluates conceptual understanding and domain reasoning using multilingual, professional exam-style financial questions (e.g., CFA, CPA, EFPA, BBF). Performance is measured by accuracy.
- **Task 2 – Multilingual Financial Q&A:** Tests analytical financial reasoning using multilingual and multimodal information sources (e.g., SEC filings plus cross-lingual news). Models generate concise, evidence-grounded answers evaluated with ROUGE and factuality metrics.
- **Task 3 – Financial Decision Making:** Assesses reasoning-to-action by generating Buy/Hold/Sell decisions and short rationales from textual and numerical market contexts (BTC, TSLA). Evaluated via profitability, stability, and risk metrics (e.g., Sharpe Ratio, Cumulative Return).

## What to Submit
- Per-task predictions in the task-specific submission format. Task 1 and Task 2 do not require confidence scores.
- Evidence trace or rationale only when required by the task definition.
- System card (model design, data usage, risks) and reproducibility notes (seed, versions, hardware).

## How to Participate
- Register via the official CLEF form (choose FinMMEval tasks): https://clef-labs-registration.dipintra.it/registrationForm.php#registrationFrom
- Task 3 endpoint submission (Agent Market Arena Google Form): https://huggingface.co/spaces/TheFinAI/Agent-Market-Arena
- Updated Task 3 endpoint submission deadline: 05 May 2026
- Full call, visuals, and timeline: https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/
- At present, we do not enforce a hard submission cap per task. Participants may submit multiple times as needed, but should avoid unnecessary rapid resubmission.

## Task 3 Endpoint Example
- A reference FastAPI endpoint implementation is available at `examples/simple_trading_api.py`.
- The example matches the documented Task 3 request/response format and returns `recommended_action` only.

## Task 3 Notes
- Task 3 uses a longer endpoint-based evaluation workflow and submitted systems may be run over an extended period through late May 2026.
- Participants are encouraged to prepare their working notes early. The paper should primarily describe the system architecture, methodology, and experimental setup; results can be updated later if the evaluation status is stated clearly.
- Awards are decided primarily based on paper quality, with leaderboard performance considered as supporting evidence.

## Training Data (Released)
- Download the training collection on Hugging Face (released 2025-12-15): https://huggingface.co/collections/MBZUAI/finmmeval-lab-clef2026
- This collection is the official public data release for the lab. Participants may use the released datasets as training resources for their systems, including reorganizing or re-splitting them as needed for model development.
- The original split names shown on individual dataset cards do not restrict participant usage.
- Task 1 dev leaderboards are based on separate organizer-held evaluation sets. Those released leaderboard dev sets are for validation only and should not be used for training. The remaining hidden test sets are reserved for final evaluation.
- Task 1 leaderboard rows marked as baselines are organizer sanity checks: Random, Always A, Round Robin, and Qwen2.5-0.5B-Instruct zero-shot.
- See each dataset card in the collection for licenses and format details.
- Task 3 historical data for backtesting, validation, and training: https://huggingface.co/collections/MBZUAI/finmmeval-lab-clef2026

## Awards
- 🏆 Best Paper Award: USD 500
- 🥈 Outstanding Paper Award ×3: USD 300 each
- 🌱 Merit / Encouragement Award ×2: USD 200 each

## Contact
- Email: zhuohan.xie@mbzuai.ac.ae
- Discord: https://discord.gg/PEMh4a2YHV

![FinMMEval Discord QR Code](docs/images/qr_fac04f98525d666b.png)

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
