# 🏷️ Product-Pricer: AI-Powered Valuation Pipeline

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)

![Llama 3](https://img.shields.io/badge/Model-Llama_3.1-blueviolet)

![OpenAI](https://img.shields.io/badge/Fine--Tuned-GPT--4o_Mini-green)

![Status](https://img.shields.io/badge/Status-Production_Ready-success)

> **Hypothesis:** Can we teach an LLM to "understand" value just by reading a product description?
>
> **Result:** Yes. Fine-tuning GPT-4o Mini achieved **96% accuracy** (within 20% margin), outperforming traditional Machine Learning by **10x**.

## 🚀 Project Overview

**Product-Pricer** is an end-to-end MLOps pipeline that predicts Amazon product prices from raw metadata. It benchmarks the evolution of pricing logic across three paradigms:

1.  **Traditional ML:** Random Forest, SVR, Linear Regression.
2.  **Frontier LLMs:** Zero-shot inference with GPT-4o, Claude 3.5, and Gemini.
3.  **Specialized Fine-Tuning:** Custom trained adapters for **Llama 3** (Open Source) and **GPT-4o Mini** (Commercial).

---

## 📊 The "Money" Charts

### 1. Accuracy Leaderboard (Lower Error is Better)

_We compared ~20 models. Fine-tuning dominated both baselines and expensive frontier models._

| Model Type                 | Top Performer    | Avg Error ($) | HIT Rate (Acc) |
| :------------------------- | :--------------- | :------------ | :------------- |
| **Fine-Tuned Commercial**  | **GPT-4o Mini**  | **$7.55**     | **96.0%**      |
| **Fine-Tuned Open Source** | **Llama 3.1 8B** | $46.06        | 74.0%          |
| Frontier Zero-Shot         | GPT-4o / Opus    | ~$35-40       | ~85%           |
| Traditional ML             | Random Forest    | $61.87        | 50.4%          |

> **HIT Rate:** Percentage of predictions within 20% of the actual price.

### 2. Visual Proof: Predicted vs. Actual

_The "Green Zone" represents accurate predictions. Notice how the Fine-Tuned model aligns tightly along the diagonal._

![Fined Fine_tuned_Llama](https://github.com/vishy04/Product-Pricer/blob/3e1d3c3dadd643474579d6f96f194babe7696fbd/results/final%2Bfine_tuned/Fine_tuned_Llama.png)

---

## 🏗️ System Architecture

This project handles the full lifecycle of data engineering and model training:

[![](https://mermaid.ink/img/pako:eNp9VtFu2zYU_RWCQAsbsNXEdiLHDwO8xG5a2EPgGHtYPBi0RDtCKVIgqTROHGCvGzAMRYoN7Uv3ULS_sP3OfmD9hF1StCzZXvVk8t577uHhIel7HIiQ4g5eSJJco_HZhCP4njxB38G8QpImkirKdcQX6IxogggPUe9WU8kJQ5dLpWmssqIReX1VmeAvHx4_md-oG5M7wW1VllAZBt2U0WV9QGZ2WlFdneDqj1n4vD89T2cZxMc_0Hm6WJiufRJQGMzWGMCLKfQUdUOSAA1VQMjJX6Yzu6BOxrnHAYlSCXBZinJx9IIvqNIR0Lya4ENvNx07aPP1bhMmJDHpV4bk219R76wLVC410WqTx4WmMyFeqWeHXgiA04jfmC4LW-pFyZLPSsADQUIqAfOf9-_-_es3dEEkYYwyF9gkKhk8S1xwymzQS5YlrFNGCbdgXz58_hu9gP1BF1IEVClRQKrYPKPvUzQWryiP7iy7arlZZLa32ILycML36Gy3BY0lifiuym7PgFPDc5mgJQgcg7NUif63YAoG6lv-jz-j4SCfKihcGfVr6PL7UXWf6E1v5iqmMdujdl8K8LOT6OPv-RjMMKeS8oAWGj2_GNdbooae0xhWtrdfy5s7hGlsF7qnZ0HVtSh9IDhO7RZYJu9_slP1bK5UbT3CSEz6Y5v66V02RE1UGYhRt1pO3VDzPQOJXBtbs4ec-WCdDvzznyhbNBrCiv8P-cibG4G1RZ4uEr0H13rla6bp3RCWWtttGcYEjCpNr5ADTgUjh2lgC4qNxnC63H4-_uJGcBCIKlpmSLWMAuVO95bPSXhDYOPDqbbF24cKVCRsRBMhddblzcYsQCuL7PNG28vzpmYlWyqVhekz8RqdCs6pXeLmWkX1-jfF26ccKF4S2W877S6CrZvRXnBzaGQMoQXKHJuFXYWpXoHScaLVKj99X0lZH6DdlJdK8MEqM9du1N3w9oJ3z8Fq7fMt2u4GKRDf9s6ap117ZoEskB_v7YDrtDNv2e7M5lS6UkdzEugSm_x9ykE9o1Cqrtfv1Mq9b6UeedJLMSsn5O2cm6GHNWHJay5mqBYcimvwmkch7swJU7SGYypjYsb43pRNsL6mMZ3gDvwM6ZykTE_whD9AXUL4D0LEuKNlCpVSpIvr9SBN4C2jZxGBIxrn4GBtMNypSLnGncPGkcXAnXt8izv1lnd82Gi2j32_5TfbjRO_Ret-DS8hdOAdmK_tt9qN45MT_7jt-wdHrYcavrMEeMpYDdMw0kIOs78o9p_Kw3-6OcW6?type=png)](https://mermaid.live/edit#pako:eNp9VtFu2zYU_RWCQAsbsNXEdiLHDwO8xG5a2EPgGHtYPBi0RDtCKVIgqTROHGCvGzAMRYoN7Uv3ULS_sP3OfmD9hF1StCzZXvVk8t577uHhIel7HIiQ4g5eSJJco_HZhCP4njxB38G8QpImkirKdcQX6IxogggPUe9WU8kJQ5dLpWmssqIReX1VmeAvHx4_md-oG5M7wW1VllAZBt2U0WV9QGZ2WlFdneDqj1n4vD89T2cZxMc_0Hm6WJiufRJQGMzWGMCLKfQUdUOSAA1VQMjJX6Yzu6BOxrnHAYlSCXBZinJx9IIvqNIR0Lya4ENvNx07aPP1bhMmJDHpV4bk219R76wLVC410WqTx4WmMyFeqWeHXgiA04jfmC4LW-pFyZLPSsADQUIqAfOf9-_-_es3dEEkYYwyF9gkKhk8S1xwymzQS5YlrFNGCbdgXz58_hu9gP1BF1IEVClRQKrYPKPvUzQWryiP7iy7arlZZLa32ILycML36Gy3BY0lifiuym7PgFPDc5mgJQgcg7NUif63YAoG6lv-jz-j4SCfKihcGfVr6PL7UXWf6E1v5iqmMdujdl8K8LOT6OPv-RjMMKeS8oAWGj2_GNdbooae0xhWtrdfy5s7hGlsF7qnZ0HVtSh9IDhO7RZYJu9_slP1bK5UbT3CSEz6Y5v66V02RE1UGYhRt1pO3VDzPQOJXBtbs4ec-WCdDvzznyhbNBrCiv8P-cibG4G1RZ4uEr0H13rla6bp3RCWWtttGcYEjCpNr5ADTgUjh2lgC4qNxnC63H4-_uJGcBCIKlpmSLWMAuVO95bPSXhDYOPDqbbF24cKVCRsRBMhddblzcYsQCuL7PNG28vzpmYlWyqVhekz8RqdCs6pXeLmWkX1-jfF26ccKF4S2W877S6CrZvRXnBzaGQMoQXKHJuFXYWpXoHScaLVKj99X0lZH6DdlJdK8MEqM9du1N3w9oJ3z8Fq7fMt2u4GKRDf9s6ap117ZoEskB_v7YDrtDNv2e7M5lS6UkdzEugSm_x9ykE9o1Cqrtfv1Mq9b6UeedJLMSsn5O2cm6GHNWHJay5mqBYcimvwmkch7swJU7SGYypjYsb43pRNsL6mMZ3gDvwM6ZykTE_whD9AXUL4D0LEuKNlCpVSpIvr9SBN4C2jZxGBIxrn4GBtMNypSLnGncPGkcXAnXt8izv1lnd82Gi2j32_5TfbjRO_Ret-DS8hdOAdmK_tt9qN45MT_7jt-wdHrYcavrMEeMpYDdMw0kIOs78o9p_Kw3-6OcW6)
