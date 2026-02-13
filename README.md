# Consumer Risk Model Mesh

Integrated agentic framework for consumer credit risk modeling with PD/LGD/EAD estimation, CCAR/DFAST stress testing, and model mesh architecture for segment-specific risk quantification.

## System Architecture

The framework implements a **layered agentic architecture** for Basel-compliant consumer credit risk modeling:

###  Layer 1: Agentic Data Processing
**Macro Agent** - Generates adverse economic scenarios (unemployment, GDP, HPI, rates)  
**Sentiment Agent** - Analyzes borrower communications for early warning signals  
**Structure Agent** - Extracts structured features from unstructured data

### Layer 2: Quantitative Risk Models
**LSTM Attention Networks** - Time-aware cash flow modeling for thin-file borrowers  
**Cox Proportional Hazards** - Baseline PD estimation with macroeconomic covariates  
**Graph Neural Networks** - Systemic contagion risk for Buy Now Pay Later (BNPL) portfolios

### Layer 3: Stress Testing Engine
**Scenario Generator** - Monte Carlo simulation with narrative-driven macro shocks  
**Sensitivity Analyzer** - Impact on capital calculations (RWA, expected loss)

### Layer 4: Calibration & Uncertainty
**Conformal Prediction** - Adaptive prediction intervals with 95% coverage guarantees  
**Bayesian Hierarchical Models** - Segment-level smoothing using PyMC

### Layer 5: Model Orchestration
**Risk Ensemble** - Combines scores across models with calibration  
**Drift Detector** - Monitors model decay vs training distribution

## Mathematical Framework

### Probability of Default (PD)

The PD model uses a **Cox Proportional Hazards** specification:

```
λ(t|X) = λ₀(t) · exp(β'·X)
```

where:
- `λ(t|X)` is the hazard rate at time t given covariates X
- `λ₀(t)` is the baseline hazard (estimated non-parametrically)
- `β` contains coefficients for borrower and macro features
- `X` includes: FICO score, DTI ratio, LTV, unemployment rate, HPI

**Time-to-default** follows:
```
S(t|X) = exp(-∫₀ᵗ λ(u|X)du) = S₀(t)^exp(β'X)
```

### Loss Given Default (LGD)

LGD is modeled using a **fractional response regression** with logit link:

```
E[LGD|X] = Λ(γ'·X) = exp(γ'X) / (1 + exp(γ'X))
```

Key drivers:
- Collateral value (LTV at origination)
- Recovery costs
- Time in collections
- Macroeconomic conditions (HPI, unemployment)

### Exposure at Default (EAD)

For revolving credit, EAD incorporates **credit conversion factor** (CCF):

```
EAD = Current_Balance + CCF × (Limit - Current_Balance)
```

**CCF estimation** uses quantile regression to capture tail behavior:
```
CCFᵗ(τ|X) = arg min ∑ ρᵗ(CCFᵢ - CCF'·Xᵢ)
```
where `ρᵗ` is the quantile check function at percentile τ (typically 75th-95th)

### Expected Loss (EL)

Basel expected loss formula:
```
EL = PD × LGD × EAD
```

### Risk-Weighted Assets (RWA)

Using the Basel IRB foundation approach:

```
RWA = K × LGD × EAD × 12.5

where K = capital requirement:

K = [LGD × N(√(1/(1-R)) × G(PD) + √(R/(1-R)) × G(0.999)) - PD × LGD] × (1 + (M-2.5) × b) / (1 - 1.5 × b)
```

**Correlation (R)** for retail exposures:
```
R = 0.03 × (1 - exp(-35 × PD)) / (1 - exp(-35)) + 
    0.16 × [1 - (1 - exp(-35 × PD)) / (1 - exp(-35))]
```

**Maturity adjustment (b)**:
```
b = [0.11852 - 0.05478 × ln(PD)]^2
```

## Model Architecture

### LSTM Attention Network

For borrowers with sparse transaction history, the model uses temporal attention:

```python
h_t = LSTM(x_t, h_{t-1})
α_t = softmax(W_a · h_t)
context = ∑ α_t · h_t
score = σ(W_o · context + b)
```

**Attention mechanism** dynamically weights payment history based on recency and macroeconomic conditions.

### Graph Neural Network (GNN) for Contagion

Models systemic risk in BNPL networks where borrowers share merchants:

```
H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
```

where:
- `A` is the adjacency matrix (borrower-merchant bipartite graph)
- `D` is the degree matrix
- `H^(l)` are node embeddings at layer l
- Message passing aggregates default signals across connected borrowers

### Conformal Prediction Intervals

Provides **distribution-free uncertainty quantification**:

```
C(X_test) = [\hat{q}_{α/2}(X_test), \hat{q}_{1-α/2}(X_test)]
```

where quantiles are calibrated on a holdout set to guarantee:
```
P(Y ∈ C(X)) ≥ 1 - α
```

for any test distribution (handles covariate shift).

## Project Structure

```
consumer-risk-model-mesh/
├── agents/              # LLM-based data preprocessing
│   ├── macro_agent.py    # Economic scenario generation
│   ├── sentiment_agent.py # Borrower communication analysis  
│   └── structure_agent.py # PDF/document parsing
├── models/              # Core risk quantification
│   ├── attention_lstm.py  # Time-aware scoring
│   ├── survival_cox.py    # Baseline PD with macro
│   └── contagion_gnn.py   # Systemic BNPL risk
├── stress_testing/      # CCAR/DFAST simulation
│   ├── scenario_generator.py
│   └── sensitivity_analyzer.py
├── calibration/         # Uncertainty quantification
│   ├── conformal.py       # Adaptive intervals
│   └── bayesian_hier.py   # Segment smoothing
├── pipeline/            # Orchestration & monitoring
│   ├── risk_ensemble.py   # Model combination
│   └── drift_detector.py  # Distribution shift detection
└── main.py              # Flask API & batch processing
```

## Key Features

### Basel III Compliance
- **IRB Foundation Approach**: PD/LGD/EAD estimation with correlation adjustments
- **CCAR/DFAST Stress Testing**: Severely adverse scenario generation
- **Procyclicality Dampening**: Through-the-cycle PD calibration
- **Conservatism Buffer**: 99.9th percentile capital calculations

### Production-Grade Architecture
- **Model Mesh**: Segment-specific model routing (prime vs subprime vs thin-file)
- **A/B Testing Framework**: Champion/challenger deployment
- **Drift Monitoring**: PSI (Population Stability Index) tracking
- **Explainability**: SHAP values for regulatory reporting

### Advanced Methodologies
- **Temporal Attention**: Captures non-linear payment patterns
- **Graph-Based Contagion**: Network effects in BNPL portfolios
- **Conformal Prediction**: Valid coverage under distribution shift
- **Bayesian Hierarchical**: Borrows strength across segments

## Technical Stack

**Core**: Python 3.9+, NumPy, Pandas, Scikit-learn  
**Deep Learning**: PyTorch, PyTorch Geometric  
**Survival Analysis**: Lifelines  
**Bayesian**: PyMC  
**Stress Testing**: Monte Carlo with scipy  
**API**: Flask, Gunicorn  
**Distributed**: PySpark (for production deployment)

## Basel Terminology Reference

**PD** - Probability of Default (12-month horizon)  
**LGD** - Loss Given Default (% of EAD lost)  
**EAD** - Exposure at Default (balance + undrawn commitment)  
**RWA** - Risk-Weighted Assets (capital requirement)  
**CCF** - Credit Conversion Factor (off-balance sheet)  
**VaR** - Value at Risk  
**CVA** - Credit Valuation Adjustment  
**Through-the-Cycle (TTC)** - Long-run average PD  
**Point-in-Time (PIT)** - Current economic conditions PD

## Stress Testing Scenarios

### Severely Adverse
- Unemployment: 4% → 10% (peak at Q3)
- GDP Growth: +2.5% → -3% (trough at Q2)
- House Prices: -25% cumulative
- VIX: 15 → 60 (spike at onset)

### Adverse
- Unemployment: 4% → 7%
- GDP Growth: +2.5% → -1.5%
- House Prices: -15% cumulative  
- VIX: 15 → 35

## Model Performance Tracking

Key metrics monitored:
- **Gini Coefficient**: Rank-ordering power
- **K-S Statistic**: Separation between goods/bads
- **Brier Score**: Calibration quality
- **PSI**: Population stability
- **Backtesting Coverage**: Conformal interval validity

---

**Note**: This framework is designed for demonstration and research purposes. Production deployment requires extensive validation, regulatory approval, and integration with enterprise risk infrastructure.
