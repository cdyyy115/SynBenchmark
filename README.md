# Systematic Evaluation of LLMs in Chemical Synthesis


*Figure 1: Benchmark and evaluation framework for Large Language Models (LLMs) in chemical synthesis tasks.*
<img width="1755" height="780" alt="image" src="https://github.com/user-attachments/assets/946c2c0e-026a-40c5-94a1-b219dd9a5cc2" />

---

## 📌 Overview

Chemical synthesis is a fundamental component of drug discovery and development, enabling the preparation of diverse molecular structures and providing the experimental basis for compound screening, lead optimization, and the development of new therapeutic agents. 

In recent years, various Large Language Models (LLMs) have been developed and increasingly applied to chemical domain tasks. However, their performance, reliability, and practical utility in chemical synthesis have not been systematically validated. 

This repository contains the benchmark code, evaluation datasets, and experimental findings from our comprehensive study evaluating both **general-purpose** and **chemistry-specific LLMs** across multiple core chemical synthesis tasks, including real-world pharmaceutical industry cases.

---

## 🔬 Benchmark Tasks

We evaluate LLMs across five core single-step/property tasks and multi-step retrosynthetic planning:

1. **Forward Reaction Prediction**: Predicting product structures from given reactants and reagents.
2. **Retrosynthetic Planning**: Deconstructing target molecules into viable starting materials.
3. **Reaction Condition Recommendation**: Suggesting optimal solvents, catalysts, temperatures, and reagents.
4. **High-Yield Reaction Identification**: Assessing and predicting reaction yield efficiency.
5. **Safety Risk Assessment**: Identifying hazardous reactions, toxic compounds, and safety protocols.
6. **Real-World Multi-Step Retrosynthesis**: Evaluating performance on **12 complex cases** derived from actual pharmaceutical company pipelines.


---
## 📜 License

This project is licensed under the [MIT License](LICENSE).
