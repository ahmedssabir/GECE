# Gender-ECE

[![Website](https://img.shields.io/badge/website-live-brightgreen)](https://ahmed.jp/project_page/cal_gece_2026/gece.html)
[![AAAI-2026 AISI Track](https://img.shields.io/badge/AAAI--2026-AISI%20Track-blue)](https://aaai.org)


This repository contains the implementation of the paper [The Confidence Trap: Gender Bias and Predictive Certainty in LLMs]().

## Overview
The increased use of Large Language Models (LLMs) in sensitive domains leads to growing interest in how their confidence scores correspond to fairness and bias. This study examines the alignment between LLM-predicted confidence and human-annotated bias judgments. Focusing on gender bias, the research investigates probability confidence calibration in contexts involving gendered pronoun resolution. The goal is to evaluate if calibration metrics based on predicted confidence scores effectively capture fairness-related disparities in LLMs. The results show that, among the six state-of-the-art models, Gemma-2 demonstrates the worst calibration according to the gender bias benchmark. The primary contribution of this work is a fairness-aware evaluation of LLMs’ confidence calibration, offering guidance for ethical deployment. In addition, we introduce a new calibration metric, Gender-ECE, designed to measure gender disparities in resolution tasks.


## Quick Start

Run `run.ipynb` to reproduce all results.


## How to Run

### Requirements
```
pip install huggingface_hub accelerate pandas transformers hf_transfer matplotlib
```
<!-- huggingface-cli login --token "your_token"-->
Run
```
python code/main.py   --dataset data/winobias.csv   --model_name Qwen/Qwen2.5-7B   
```


## Citation

The details of this repo are described in the following paper. If you find this repo useful, please kindly cite it:

```bibtex
@article{sabir2026confidence,
  title={The Confidence Trap: Gender Bias and Predictive Certainty in LLMs},
  author={Sabir, Ahmed and K{\"a}ngsepp, Markus and Sharma, Rajesh},
  journal={arXiv preprint arXiv:2601.07806},
  year={2026}
}
```

<!--
### Acknowledgement
This work was supported by the Estonian Research Council grant “Developing human-centric digital solutions” (TEM TA120) and by the Estonian Centre of Excellence in Artificial Intelligence (EXAI), funded by the Estonian Ministry of Education and Research and co-funded by the European Union and the Estonian Research Council via project TEM TA119. It was funded by the EU H2020 program under the SoBigData++ project (grant agreement No. 871042) and partially funded by the HAMISON project.
-->

<!--
### Acknowledgement
This work was supported by the Estonian Research Council grant “Developing human-centric digital solutions” (TEM TA120) and by the Estonian Centre of Excellence in Artificial Intelligence (EXAI), funded by the Estonian Ministry of Education and Research and co-funded by the European Union and the Estonian Research Council via project TEM TA119. It was funded by the EU H2020 program under the SoBigData++ project (grant agreement No. 871042) and partially funded by the HAMISON project.
-->
