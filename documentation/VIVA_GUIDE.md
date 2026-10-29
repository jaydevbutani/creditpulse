# CreditPulse — Viva & Faculty Defense Master Guide

> Concise viva examination questions and answers. Each response is formulated in 1–3 clear, high-yield sentences.

---

### 1. What problem does CreditPulse solve?
Traditional credit scoring models rely on static annual financial ratios or point-in-time bureau scores, which miss month-to-month liquidity deterioration. CreditPulse acts as an early-warning distress monitor by analyzing continuous monthly account transactions to detect borrower distress months before formal default occurs.

### 2. What is the novelty?
The primary novelty is **Human-in-the-Loop Early Warning**: the system cleanly separates algorithmic neural alerts from independent human examiner flags, making model–human disagreements visible and auditable. The second novelty is **temporal sequence modeling with dual multi-task heads**, predicting both overall 6-month distress probability and the near-term monthly hazard timing curve.

### 3. What data goes into the model?
The model receives a sequence of the borrower's monthly account records comprising 44 documented behavioral features covering liquidity reserves, net cash flows, overdraft frequency, and debt-service ratios. Every feature position is populated with continuous financial formulas without zero-padding.

### 4. Why 12 months?
A 12-month sequence captures an entire annual operating cycle, accounting for business seasonality, quarterly tax deadlines, and working capital fluctuations. It provides sufficient historical context to distinguish temporary timing dips from chronic financial distress.

### 5. Why Transformer?
Self-attention computes pairwise correlations across all 12 temporal tokens in $O(1)$ path length, allowing the model to directly link early warning signals (like a credit line reduction in month 2) with subsequent distress (in month 10) without the vanishing gradient problems of LSTMs or the information loss of summary statistics.

### 6. What is M6_multitask?
`M6_multitask` is our proposed Pre-LayerNorm Transformer featuring dual prediction heads: Head A predicts 6-month binary distress, while Head B predicts a 6-month discrete hazard vector. Multi-task learning regularizes internal representations, preventing overfitting on sparse default labels.

### 7. How many layers?
The model has **3 stacked Pre-LayerNorm Transformer encoder blocks**. Normalizing inputs before self-attention and MLP sublayers ensures gradient stability during AdamW optimization.

### 8. How many attention heads?
The model uses **2 multi-head self-attention heads** with a dimension of $d_k = 16$ per head, totaling $d_{\text{model}} = 32$. This allows one head to focus on short-term liquidity shocks while the other tracks medium-term cash flow trends.

### 9. What activation function?
We use **GELU (Gaussian Error Linear Unit)** in the feedforward sublayers. GELU avoids the dying-neuron problem of standard ReLU by smoothly gating values according to their probabilistic distribution.

### 10. What loss function?
We optimize a combined **Multi-Task Loss**:
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{HeadA}} + 0.3 \times \mathcal{L}_{\text{HeadB}}$$
Head A uses positive-weighted Binary Cross-Entropy with `pos_weight = 8.62` to address class imbalance, and Head B uses discrete-time hazard cross-entropy.

### 11. What optimizer?
We use **AdamW** with a learning rate of $0.0002$ and weight decay of $0.0001$. AdamW decouples weight decay from gradient updates, providing superior generalization compared to standard Adam.

### 12. How is the model evaluated?
The model is evaluated strictly **Out-of-Time (OOT)** on the 1998 test set split of the Czech banking panel, completely avoiding lookahead leakage. Performance is measured using PR-AUC, ROC-AUC, Brier score, and Capture @ Top 10% queue.

### 13. Why PR-AUC?
Because loan default is a severe imbalanced event (~4.2% prevalence). In rare-event problems, standard ROC-AUC is artificially inflated by the large volume of true negatives, whereas Precision-Recall AUC (PR-AUC) directly reflects the precision of the high-risk review queue.

### 14. What does the human flag do?
A human flag allows a credit officer to independently mark a loan for review based on qualitative information the model cannot see, such as management disputes, pending litigation, or local industry headwinds. It creates an auditable review record without altering neural weights.

### 15. How is human feedback different from model prediction?
Model alerts are purely statistical derivations from monthly account transactions, whereas human flags reflect supervisory discretion. The system highlights **Model–Human Disagreements**, ensuring the institution never blindly trusts automated AI predictions.

### 16. What happens when a new bank record arrives?
When a transaction is processed, it is validated and stored in SQLite, the latest 12-month feature matrix is constructed, the M6 Transformer computes an updated risk probability, and the system displays the previous vs new risk alongside the primary behavioral reasons for the change.
