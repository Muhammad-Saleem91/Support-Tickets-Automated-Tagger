# 🏷️ Task 5: Auto-Tagging Support Tickets with LLMs

## 📌 Objective
The goal of this task was to build an automated classification system for customer support tickets using **Large Language Models (LLMs)**. Instead of training a model from scratch (which requires extensive labeled data), we explored **In-Context Learning** techniques to categorize tickets into specific queues (e.g., *Billing, Technical Support, Returns*).

The project specifically compares two approaches:
1.  **Zero-Shot Learning:** Classifying text without any prior examples.
2.  **Few-Shot Learning:** Using Prompt Engineering to "teach" the model via examples.

## 🛠️ Methodology / Approach

### 1. Dataset & Preprocessing
* **Source:** Real-world multi-language support ticket dataset (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`).
* **Filtering:** Filtered for **English ('en')** tickets to ensure optimal compatibility with standard pre-trained models.
* **Context Engineering:** Concatenated the `Subject` and `Body` of the email to provide the model with full context.
* **Truncation:** Limited input text to 512 characters to fit within the model's context window and ensure efficient processing.

### 2. Model Architectures
We utilized the Hugging Face `transformers` library to implement two distinct pipelines:

* **Approach A: Zero-Shot Classification**
    * **Model:** `facebook/bart-large-mnli`
    * **Technique:** Natural Language Inference (NLI).
    * **Logic:** The model treats the classification as a hypothesis test (e.g., *Premise: "My screen is broken"*, *Hypothesis: "This is a Hardware issue"*). It selects the tag with the highest entailment probability.

* **Approach B: Few-Shot Prompting**
    * **Model:** `google/flan-t5-base` (Seq2Seq Model)
    * **Technique:** Prompt Engineering.
    * **Logic:** We constructed a custom prompt containing **3 labeled examples** ("shots") to guide the model's generation. This helps the model understand the specific mapping between customer complaints and company-specific queue names.

## 📊 Key Results & Observations

### Performance Comparison
| Metric | Zero-Shot (BART) | Few-Shot (FLAN-T5) |
| :--- | :--- | :--- |
| **Accuracy** | ~40-60% (Baseline) | **Higher (via Prompting)** |

### Observations
1.  **Zero-Shot Flexibility:** This approach is excellent for "Cold Start" scenarios. It correctly identified general categories (e.g., *Billing*) but struggled with company-specific jargon or overlapping categories (e.g., distinguishing *Product Support* from *Technical Support*).
2.  **Power of Prompt Engineering:** The Few-Shot model generally performed better because the prompt acted as a "style guide." By showing the model that *"Login failed"* belongs to *Technical Support*, it learned the specific classification rules without any weight updates.
3.  **Efficiency:** Both models ran efficiently on a GPU, processing 50 tickets in under a minute, demonstrating the viability of LLMs for real-time ticket tagging.

## 🚀 Files Included
* `Task5_Final_Submission.ipynb`: The complete Python notebook containing the pipeline code.
* `task5_final_results.csv`: A CSV file containing the side-by-side predictions of both models against the ground truth.
* `requirements.txt`: List of dependencies required to run the project.