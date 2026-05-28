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
- Updated Task 3 endpoint submission deadline: 10 May 2026 AoE
- Full call, visuals, and timeline: https://mbzuai-nlp.github.io/CLEF-2026-FinMMEval-Lab/
- At present, we do not enforce a hard submission cap per task. Participants may submit multiple times as needed, but should avoid unnecessary rapid resubmission.
- Task 1/2 dev leaderboards and test questions release: 06 May 2026.
- Task 1/2 final run submission deadline: 25 May 2026 AoE.
- Task 1/2 final leaderboard release: 27 May 2026.

## Task 3 Endpoint Example
- A reference FastAPI endpoint implementation is available at `examples/simple_trading_api.py`.
- The example matches the documented Task 3 request/response format and returns `recommended_action` only.

## Working Notes Template
- Participant working notes should follow the CLEF Labs Working Notes format.
- Working notes are published in CEUR-WS proceedings and should use the CLEF 2026 1-column CEURART working notes template.
- Papers should be written in English, with a minimum length of 5 pages and no maximum page limit.
- Paper titles must clearly include the team/system name, `FinMMEval 2026`, and the task number(s) covered by the paper. This is important for EasyChair metadata, review handling, proceedings grouping, and the CEUR-WS table of contents.
- Recommended title patterns:
  - `<Team Name> @ FinMMEval 2026 Task <X>: <Short System Description>`
  - `<Team Name> @ FinMMEval 2026: Systems for Tasks 1 and 2`
  - If the team name already contains `@`, for example `DS@GT`, `DS@GT at FinMMEval 2026 Task 2: <Short System Description>` is also acceptable.
- Please make sure the title in EasyChair exactly matches the title in the PDF.
- CLEF 2026 LaTeX template: https://clef2026.clef-initiative.eu/calls/submitting/clef26-working-notes-template.zip
- CLEF 2026 ODT template: https://clef2026.clef-initiative.eu/calls/submitting/clef26-working-notes-template.odt
- CLEF 2026 instructions PDF: https://clef2026.clef-initiative.eu/calls/submitting/clef26-working-notes-instructions.pdf
- Generic CEURART Overleaf template, for reference: https://www.overleaf.com/latex/templates/template-for-submissions-to-ceur-workshop-proceedings-ceur-ws-dot-org/wqyfdgftmcfw
- CLEF 2026 submission instructions: https://clef2026.clef-initiative.eu/calls/submitting/

## Task 3 Notes
- The Task 3 endpoint submission deadline is the deadline for submitting or updating an endpoint; it is not the end of the evaluation period.
- Submitting the Google Form registers or updates an endpoint, but it may not appear on the leaderboard immediately. Organizers first verify submitted endpoints and confirm the final endpoint list.
- Official Task 3 performance is computed over a common evaluation window for all accepted endpoints, rather than starting separately from each team's individual form submission date.
- Task 3 uses a longer endpoint-based evaluation workflow. Submitted systems will continue to be called daily after the endpoint submission deadline for the official Task 3 evaluation window. We expect this window to run through late June or early July, aligned with the final lab reporting schedule.
- The daily runner starts at 00:00 UTC. Teams do not need to keep endpoints online for the full day, but should start them shortly before 00:00 UTC and keep them available for several hours to allow for queued requests, retries, and temporary network delays.
- Participants are encouraged to prepare their working notes early. The paper should primarily describe the system architecture, methodology, and experimental setup; results can be updated later if the evaluation status is stated clearly.
- Awards are decided primarily based on paper quality, with leaderboard performance considered as supporting evidence.

## Working Notes and Awards
- FinMMEval participant systems should be documented in CLEF Working Notes papers.
- FinMMEval awards are evaluated over FinMMEval participant submissions and their CLEF Working Notes papers.
- The separate CLEF main conference paper track is not used for FinMMEval award evaluation.
- Participants should not submit the same or near-identical manuscript to both the CLEF main conference track and the CLEF Working Notes track.
- After the lab, a substantially extended version may be submitted to another conference or journal if it follows that venue's originality, prior-publication, and dual-submission policies.

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

## Recommended Citations

Please cite the FinMMEval Lab overview paper when referring to the full lab. If your work is associated with a specific task, please also cite the corresponding task overview paper.

In addition, please cite the task source or platform papers used by the tracks you participated in:

- Task 1 English / Chinese: cite RealFin.
- Task 1 Hindi: cite BhashaBench V1 / BhashaBench-Finance.
- Task 1 Arabic: cite SAHM.
- Task 2: cite MultiFinBen.
- Task 3: cite Agent Market Arena / When Agents Trade.

Some source papers have been accepted but are not yet available in the official proceedings. We provide the arXiv citations below for now and will update them to the official proceedings BibTeX once available.

```bibtex
@inproceedings{FinMMEval2026,
  title = {Overview of {FinMMEval} 2026: Multilingual and Multimodal Financial Evaluation},
  author = {Zhuohan Xie and Yuyang Dai and Rania Elbadry and Vanshikaa Jani and Xueqing Peng and Lingfei Qian and Georgi Georgiev and Dimitar Dimitrov and Fan Zhang and Jimin Huang and Jiahui Geng and Yankai Chen and Ye Yuan and Haolun Wu and Yuxia Wang and Ivan Koychev and Veselin Stoyanov and Mingzi Song and Yu Chen and Steve Liu and Preslav Nakov},
  booktitle = {Experimental IR Meets Multilinguality, Multimodality, and Interaction},
  series = {Proceedings of the Seventeenth International Conference of the CLEF Association (CLEF 2026)},
  year = {2026},
  month = {September 21--24},
  address = {Jena, Germany},
  publisher = {Springer Lecture Notes in Computer Science LNCS},
}

@inproceedings{FinMMEvalTask1Overview2026,
  title = {Overview of the {FinMMEval} 2026 Task 1: Multilingual Financial Multiple-Choice Question Answering},
  author = {Zhuohan Xie and Yuyang Dai and Rania Elbadry and Vanshikaa Jani and Georgi Georgiev and Dimitar Dimitrov and Fan Zhang and Xueqing Peng and Lingfei Qian and Jimin Huang and Jiahui Geng and Yankai Chen and Ye Yuan and Haolun Wu and Yuxia Wang and Ivan Koychev and Veselin Stoyanov and Mingzi Song and Yu Chen and Steve Liu and Preslav Nakov},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  year = {2026},
  month = {September 21--24},
  address = {Jena, Germany},
  publisher = {CEUR-WS.org},
}

@inproceedings{FinMMEvalTask2Overview2026,
  title = {Overview of the {FinMMEval} 2026 Task 2: Financial Question Answering and Summarization},
  author = {Zhuohan Xie and Xueqing Peng and Georgi Georgiev and Dimitar Dimitrov and Rania Elbadry and Fan Zhang and Lingfei Qian and Jimin Huang and Vanshikaa Jani and Yuyang Dai and Jiahui Geng and Yankai Chen and Ye Yuan and Haolun Wu and Yuxia Wang and Ivan Koychev and Veselin Stoyanov and Mingzi Song and Yu Chen and Steve Liu and Preslav Nakov},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  year = {2026},
  month = {September 21--24},
  address = {Jena, Germany},
  publisher = {CEUR-WS.org},
}

@inproceedings{FinMMEvalTask3Overview2026,
  title = {Overview of the {FinMMEval} 2026 Task 3: Financial Decision Making},
  author = {Zhuohan Xie and Lingfei Qian and Georgi Georgiev and Dimitar Dimitrov and Rania Elbadry and Fan Zhang and Xueqing Peng and Jimin Huang and Vanshikaa Jani and Yuyang Dai and Jiahui Geng and Yankai Chen and Ye Yuan and Haolun Wu and Yuxia Wang and Ivan Koychev and Veselin Stoyanov and Mingzi Song and Yu Chen and Steve Liu and Preslav Nakov},
  booktitle = {CLEF 2026 Working Notes},
  series = {CEUR Workshop Proceedings},
  year = {2026},
  month = {September 21--24},
  address = {Jena, Germany},
  publisher = {CEUR-WS.org},
}

@misc{dai2026realfin,
  title = {{RealFin}: How Well Do {LLM}s Reason About Finance When Users Leave Things Unsaid?},
  author = {Yuyang Dai and Yan Lin and Zhuohan Xie and Yuxia Wang},
  year = {2026},
  eprint = {2602.07096},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.ST},
  doi = {10.48550/arXiv.2602.07096},
  url = {https://arxiv.org/abs/2602.07096},
}

@misc{devane2025bhashabenchv1,
  title = {{BhashaBench V1}: A Comprehensive Benchmark for the Quadrant of Indic Domains},
  author = {Vijay Devane and Mohd Nauman and Bhargav Patel and Aniket Mahendra Wakchoure and Yogeshkumar Sant and Shyam Pawar and Viraj Thakur and Ananya Godse and Sunil Patra and Neha Maurya and Suraj Racha and Nitish Kamal Singh and Ajay Nagpal and Piyush Sawarkar and Kundeshwar Vijayrao Pundalik and Rohit Saluja and Ganesh Ramakrishnan},
  year = {2025},
  eprint = {2510.25409},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2510.25409},
}

@misc{elbadry2026sahm,
  title = {{SAHM}: A Benchmark for Arabic Financial and Shari'ah-Compliant Reasoning},
  author = {Rania Elbadry and Sarfraz Ahmad and Ahmed Heakl and Dani Bouch and Momina Ahsan and Muhra AlMahri and Marwa Elsaid khalil and Yuxia Wang and Salem Lahlou and Sophia Ananiadou and Veselin Stoyanov and Jimin Huang and Xueqing Peng and Preslav Nakov and Zhuohan Xie},
  year = {2026},
  eprint = {2604.19098},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2604.19098},
  url = {https://arxiv.org/abs/2604.19098},
}

@misc{peng2025multifinben,
  title = {{MultiFinBen}: Benchmarking Large Language Models for Multilingual and Multimodal Financial Application},
  author = {Xueqing Peng and Lingfei Qian and Yan Wang and Ruoyu Xiang and Yueru He and Yang Ren and Mingyang Jiang and Vincent Jim Zhang and Yuqing Guo and Jeff Zhao and Huan He and Yi Han and Yun Feng and Yuechen Jiang and Yupeng Cao and Haohang Li and Yangyang Yu and Xiaoyu Wang and Penglei Gao and Shengyuan Lin and Keyi Wang and Shanshan Yang and Yilun Zhao and Zhiwei Liu and Peng Lu and Jerry Huang and Suyuchen Wang and Triantafillos Papadopoulos and Polydoros Giannouris and Efstathia Soufleri and Nuo Chen and Zhiyang Deng and Heming Fu and Yijia Zhao and Mingquan Lin and Meikang Qiu and Kaleb E Smith and Arman Cohan and Xiao-Yang Liu and Jimin Huang and Guojun Xiong and Alejandro Lopez-Lira and Xi Chen and Junichi Tsujii and Jian-Yun Nie and Sophia Ananiadou and Qianqian Xie},
  year = {2025},
  eprint = {2506.14028},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2506.14028},
  url = {https://arxiv.org/abs/2506.14028},
}

@misc{qian2025whenagentstrade,
  title = {When Agents Trade: Live Multi-Market Trading Benchmark for {LLM} Agents},
  author = {Lingfei Qian and Xueqing Peng and Yan Wang and Vincent Jim Zhang and Huan He and Hanley Smith and Yi Han and Yueru He and Haohang Li and Yupeng Cao and Yangyang Yu and Alejandro Lopez-Lira and Peng Lu and Jian-Yun Nie and Guojun Xiong and Jimin Huang and Sophia Ananiadou},
  year = {2025},
  eprint = {2510.11695},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2510.11695},
  url = {https://arxiv.org/abs/2510.11695},
}
```

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
- Ye Yuan (McGill University, Canada & Mila - Quebec AI Institute, Canada)
- Haolun Wu (McGill University, Canada & Mila - Quebec AI Institute, Canada)
- Yuxia Wang (INSAIT, Bulgaria)
- Ivan Koychev (Sofia University "St. Kliment Ohridski", Bulgaria)
- Veselin Stoyanov (MBZUAI, UAE)
- Mingzi Song (Nikkei Financial Technology Research Institute, Inc., Japan)
- Yu Chen (The University of Tokyo, Japan)
- Steve Liu (McGill University, Canada & MBZUAI, UAE)
- Preslav Nakov (MBZUAI, UAE)

Questions? Reach out to the organizers listed above or via the website.
