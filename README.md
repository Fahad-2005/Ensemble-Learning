# Manual Ensemble Learning & Generalization Framework

A comprehensive, production-grade machine learning framework built entirely from scratch to implement, evaluate, and benchmark Ensemble Learning algorithms. This project avoids higher-level machine learning APIs (such as scikit-learn classifiers) to explore the core mathematical mechanics, bias-variance trade-offs, and structural behaviors of ensemble variants when exposed to adversarial data distributions.

---

## 🔥 Key Architectural Features

* **Custom C4.5 Decision Tree Base:** Reuses a scratch-built binary decision tree optimizing splits via Entropy, Information Gain, and Split Information normalized into the **Gain Ratio**. Includes programmatic hooks for continuous feature parsing and split-node subspace sampling.
* **Manual Row Bootstrapping (Bagging):** Implements uniform row sampling with replacement to generate ensemble diversity and decouple variance errors among unconstrained estimators.
* **Random Forest with OOB Tracking:** Introduces node feature randomization ($\lfloor\sqrt{d}\rfloor$) to mitigate feature dominance and breaks down generalizations via a custom, built-in **Out-of-Bag (OOB) Error** estimation pipeline.
* **Sequential Boosting (AdaBoost):** Implements iterative error correction utilizing custom single-split **Decision Stumps**, tracking sample weight evolutions, and processing final predictions via weighted voting metrics ($\alpha$).

---

## 📊 Performance Benchmarks & Empirical Observations

The following execution metrics were logged directly from the scratch-built models utilizing the reproducibility dataset seeds:

* **Bagging Classifier:** Achieved a **Peak Validation Accuracy of 59.83%** on highly volatile datasets injected with 10% label noise, showing steady variance reduction as the base estimator scale expanded up to 15 trees.
* **Random Forest Classifier:** Successfully navigated skewed datasets, returning a **Manual OOB Accuracy Score of 91.86%** alongside an explicit **Validation Score of 92.61%** on severe class-imbalanced data topologies.
* **AdaBoost Classifier:** Demonstrated clean sequential convergence but highlighted standard boosting noise sensitivity patterns across explicitly controlled label noise vectors:
    * *5% Managed Noise Profile:* **80.56%** Validation Accuracy
    * *15% Moderate Noise Profile:* **72.22%** Validation Accuracy
    * *30% Aggressive Noise Profile:* **62.50%** Validation Accuracy

---

## ⚙️ Project Installation & Execution Guide

### 1. Prerequisites

Ensure you have a standard Python 3.x environment set up. You must install the necessary numeric parsing and visualization utilities before launching the pipeline:
```bash
pip install numpy pandas matplotlib scikit-learn

2. Workspace Directory Setup
To execute the modules seamlessly, maintain the following local repository file layout:
├── dataset_generator.py      # Programmatic synthesis of validation data profiles
├── decision_tree_base.py     # C4.5 base node tree module 
├── bagging.py                # Row bootstrapping parallel ensemble structure
├── random_forest.py          # Feature-subspace variance reducer and OOB engine
├── adaboost.py               # Sequential weight update optimization model
└── main.py                   # Master orchestration engine and plotting controller

3. Execution Pipeline
Open your system terminal, navigate directly into the root workspace folder, and trigger the primary script:
python main.py

4. Verification of Output Artifacts
Once execution completes, the engine will dump two analytical metric visualizations directly into your root directory:

bagging_evaluation.png: Line graphs tracking structural error paths across changing tree scales.

adaboost_noise_degradation.png: Bar chart mapping model breakdown when handling high label-noise ratios.