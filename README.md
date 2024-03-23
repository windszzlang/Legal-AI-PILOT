# Legal-AI-PILOT

Code and data repo for the paper "[PILOT: Legal Case Outcome Prediction with Case Law](https://arxiv.org/abs/2401.15770)",

Accepted at NAACL2024 Main Conference

Machine learning shows promise in predicting the outcome of legal cases, but most research has concentrated on civil law cases rather than case law systems. We identified two unique challenges in making legal case outcome predictions with case law. First, it is crucial to identify relevant precedent cases that serve as fundamental evidence for judges during decision-making. Second, it is necessary to consider the evolution of legal principles over time, as early cases may adhere to different legal contexts. In this paper, we proposed a new framework named PILOT (PredictIng Legal case OuTcome) for case outcome prediction. It comprises two modules for relevant case retrieval and temporal pattern handling, respectively. To benchmark the performance of existing legal case outcome prediction models, we curated a dataset from a large-scale case law database. We demonstrate the importance of accurately identifying precedent cases and mitigating the temporal shift when making predictions for case law, as our method shows a significant improvement over the prior methods that focus on civil law case outcome predictions.


## ECHR2023 Dataset

## Quick Start

### Code

### Dependencies

### Run


## Cite us

If you find this repo useful, please cite the following paper:

```
@misc{cao2024pilot,
      title={PILOT: Legal Case Outcome Prediction with Case Law}, 
      author={Lang Cao and Zifeng Wang and Cao Xiao and Jimeng Sun},
      year={2024},
      eprint={2401.15770},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
