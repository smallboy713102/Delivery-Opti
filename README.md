# Delivery Prediction and Resource Optimization

An end-to-end **predictive and prescriptive analytics** project for estimating last-mile delivery times, identifying orders at risk of delay, and optimizing vehicle assignment and order batching using a **Mixed-Integer Linear Program (MILP)**.

The project was developed using a public Amazon delivery dataset containing more than **43,000 last-mile orders**.

---

## Project Overview

Last-mile delivery operations involve two connected decisions:

1. **Prediction:** How long is an order expected to take?
2. **Optimization:** Which orders should be delivered individually or batched together, and which vehicle should be assigned?

This project connects both stages:

```text
Data cleaning
    ↓
Feature engineering
    ↓
KNN preprocessing
    ↓
Delivery-time prediction
    ↓
Historical SLA estimation
    ↓
Late-delivery risk classification
    ↓
Candidate batch generation
    ↓
MILP trip and vehicle selection
```

---

## Business Objective

The objective is to reduce delivery-resource usage while controlling predicted service delays.

The optimization model attempts to:

- reduce the number of delivery trips,
- reduce total route distance,
- reduce simulated operating cost,
- assign an appropriate vehicle,
- batch only geographically and temporally compatible orders,
- ensure every eligible order is assigned exactly once, and
- avoid batches that generate excessive predicted lateness.

---

## Dataset

The dataset contains **43,739 orders** and the following information:

- order ID,
- agent age and rating,
- store latitude and longitude,
- customer latitude and longitude,
- order date and time,
- pickup time,
- weather,
- traffic condition,
- vehicle type,
- delivery area,
- product category, and
- actual delivery time.

> The dataset does not include official Amazon SLA commitments, real fleet costs, labour costs, vehicle availability, or rider schedules. Therefore, SLA and cost values used in this project are data-driven or simulated assumptions.

---

## Feature Engineering

### Temporal features

The project creates:

- order time in minutes,
- pickup time in minutes,
- preparation delay,
- order hour,
- weekend indicator,
- peak-hour indicator, and
- chronological order timestamp.

Preparation delay is calculated as:

\[
\text{Preparation Delay}
=
\text{Pickup Time}
-
\text{Order Time}
\]

Overnight orders are corrected by adding 1,440 minutes when necessary.

### Geographic distance

Store-to-customer distance is calculated using the Haversine formula:

\[
d =
2R\arcsin\left(
\sqrt{
\sin^2\left(\frac{\Delta\phi}{2}\right)
+
\cos(\phi_1)\cos(\phi_2)
\sin^2\left(\frac{\Delta\lambda}{2}\right)
}
\right)
\]

where:

- \(R = 6371.0088\) km,
- \(\phi\) is latitude, and
- \(\lambda\) is longitude.

Additional indicators identify missing or invalid geographical information.

---

## Missing-Value Treatment

The project uses **K-Nearest Neighbour imputation** for both numerical and categorical predictors.

### Numerical variables

Numerical variables are standardized before KNN imputation:

\[
z_{ij}
=
\frac{x_{ij}-\mu_j}{\sigma_j}
\]

A missing value is estimated using seven distance-weighted neighbours:

\[
\hat{x}_{ij}
=
\frac{
\sum_{r\in N_7(i)} w_{ir}x_{rj}
}{
\sum_{r\in N_7(i)} w_{ir}
}
\]

where:

\[
w_{ir}
=
\frac{1}{d(i,r)}
\]

### Categorical variables

Categorical variables are:

1. ordinally encoded,
2. KNN-imputed,
3. rounded to the nearest category code, and
4. one-hot encoded.

All preprocessing transformations are fitted only on the training period.

---

## Chronological Validation

The data is sorted by order timestamp.

- Earliest 80%: training set
- Latest 20%: test set

This provides a more realistic evaluation than a random split because future orders are predicted using historical orders.

```text
Training orders: 34,991
Test orders:      8,748
```

---

## Delivery-Time Models

The following models are benchmarked:

- Mean baseline
- Linear Regression
- Random Forest Regression
- Histogram Gradient Boosting Regression

### Evaluation metrics

#### Mean Absolute Error

\[
MAE
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
\]

#### Root Mean Squared Error

\[
RMSE
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
\]

#### Coefficient of determination

\[
R^2
=
1-
\frac{
\sum_i(y_i-\hat{y}_i)^2
}{
\sum_i(y_i-\bar{y})^2
}
\]

### Model results

| Model | Test MAE | Test RMSE | Test \(R^2\) |
|---|---:|---:|---:|
| Histogram Gradient Boosting | **17.66 min** | **22.77 min** | **0.811** |
| Random Forest | 18.75 min | 23.85 min | 0.793 |
| Linear Regression | 25.38 min | 32.20 min | 0.622 |
| Mean baseline | 41.68 min | 52.42 min | approximately 0 |

Histogram Gradient Boosting reduced MAE by approximately:

\[
\frac{41.68-17.66}{41.68}\times100
\approx 57.6\%
\]

relative to the mean baseline.

---

## Historical SLA

**SLA** stands for **Service Level Agreement**.

The dataset does not provide an official promised-delivery SLA. Therefore, this project constructs a **historical delivery-time benchmark**.

For a historical segment \(g\):

\[
SLA_g
=
Q_{0.75}
\left(
\text{Delivery Time}\mid g
\right)
\]

The SLA is the 75th percentile of delivery time among similar historical training orders.

Similarity is based on exact operational grouping using a fallback hierarchy:

1. distance band + area + traffic + vehicle + category,
2. distance band + area + traffic + vehicle,
3. distance band + traffic + vehicle,
4. distance band + traffic,
5. traffic + area,
6. traffic, and
7. global training SLA.

A segment is used only when it contains at least 50 historical orders.

> These values are historical benchmarks and must not be interpreted as official Amazon contractual SLAs.

---

## Late-Delivery Risk Classification

Let:

- \(\hat{T}_i\) be predicted delivery time,
- \(SLA_i\) be the historical SLA, and
- \(MAE = 17.66\) minutes.

The risk rules are:

\[
Risk_i=
\begin{cases}
\text{Likely Late},
& \hat{T}_i>SLA_i \\[4pt]
\text{At Risk},
& SLA_i-MAE<\hat{T}_i\leq SLA_i \\[4pt]
\text{On-Time},
& \hat{T}_i\leq SLA_i-MAE
\end{cases}
\]

### Risk-layer evaluation

| Metric | Result |
|---|---:|
| Precision | 0.699 |
| Recall | 0.634 |
| F1 score | 0.665 |
| ROC-AUC | 0.900 |

The actual late-delivery rates were:

| Risk class | Actual late rate |
|---|---:|
| On-Time | 4.27% |
| At Risk | 28.63% |
| Likely Late | 69.92% |

---

## Vehicle Assumptions

The optimization uses simulated fleet parameters because actual operational values are unavailable.

| Vehicle | Capacity | Speed | Fixed cost | Cost/km | Maximum route |
|---|---:|---:|---:|---:|---:|
| Bicycle | 1 | 15 km/h | 40 | 2 | 5 km |
| Scooter | 2 | 25 km/h | 55 | 4 | 18 km |
| Motorcycle | 3 | 30 km/h | 65 | 5 | 35 km |
| Van | 6 | 22 km/h | 100 | 12 | Unlimited |

The current implementation allows batches of up to:

```python
max_batch = 3
```

Therefore:

- individual delivery: 1 order,
- two-order batch: 2 orders,
- three-order batch: 3 orders.

---

## Batch Candidate Generation

Orders are considered compatible when they satisfy operational constraints such as:

\[
|\text{Order Time}_i-\text{Order Time}_j|
\leq 20\text{ minutes}
\]

\[
d(\text{Store}_i,\text{Store}_j)
\leq 1\text{ km}
\]

\[
d(\text{Drop}_i,\text{Drop}_j)
\leq 3\text{ km}
\]

Additional conditions include:

- no order in the batch is already classified as Likely Late,
- vehicle capacity supports the batch,
- route distance is within the vehicle limit,
- batch delivery times remain within the allowed SLA tolerance, and
- the batch costs less than delivering the orders separately.

For three-order batching, compatible triples are generated using a compatibility graph rather than checking every possible triple.

---

## Batch-Time Calculation

For order \(i\) in batch \(b\):

\[
T_{ib}
=
\hat{T}_{iv}
+
W_{ib}
+
E_{ib}
\]

where:

- \(\hat{T}_{iv}\) is the vehicle-specific ML prediction,
- \(W_{ib}\) is waiting time for all batched orders to become ready, and
- \(E_{ib}\) is the incremental route-detour delay.

Waiting time is:

\[
W_{ib}
=
\max(
0,
\text{Dispatch Time}_b-\text{Pickup Time}_i
)
\]

Incremental route delay is:

\[
E_{ib}
=
\max(
0,
\text{Route Arrival}_{ib}
-
\text{Direct Travel}_{iv}
)
\]

Only incremental batching delay is added. The complete route time is not added again because distance, traffic, and vehicle information are already represented in the ML prediction.

Predicted late minutes are:

\[
L_{ib}
=
\max(
0,
T_{ib}-SLA_i
)
\]

---

## MILP Formulation

The optimization problem is a **set-partitioning MILP**.

For every candidate trip \(b\):

\[
x_b=
\begin{cases}
1,& \text{candidate trip }b\text{ is selected}\\
0,& \text{otherwise}
\end{cases}
\]

### Candidate cost

\[
C_b
=
F_v
+
k_vD_b
+
\lambda
\sum_{i\in b} L_{ib}
\]

where:

- \(F_v\) is fixed vehicle cost,
- \(k_v\) is cost per kilometre,
- \(D_b\) is route distance,
- \(L_{ib}\) is predicted late time, and
- \(\lambda\) is the lateness penalty.

### Objective

\[
\min
\sum_{b\in B}
C_bx_b
\]

### Exact assignment constraint

Every eligible order must appear in exactly one selected trip:

\[
\sum_{b:i\in b}x_b=1
\qquad \forall i
\]

### Binary condition

\[
x_b\in\{0,1\}
\]

The optimization is solved using SciPy's MILP interface with the HiGHS solver.

---

## MILP Results

The updated three-order batching model optimized **8,186 eligible orders**.

### Selected trip plan

| Trip type | Selected trips |
|---|---:|
| Individual | 7,253 |
| Two-order batch | 438 |
| Three-order batch | 19 |
| **Total optimized trips** | **7,710** |

Coverage check:

\[
7253+2(438)+3(19)=8186
\]

Therefore, every eligible order is assigned exactly once.

### Policy impact

| Metric | Result |
|---|---:|
| Orders optimized | 8,186 |
| Orders batched | 933 |
| Baseline trips | 8,186 |
| Optimized trips | 7,710 |
| Trips saved | 476 |
| Trip reduction | **5.82%** |
| Distance reduction | **4.75%** |
| Simulated cost reduction | **4.03%** |

> Distance and cost reductions are simulated results under the stated vehicle and penalty assumptions.

---

## Route Visualization

Graphviz is used to display:

- selected store pickups,
- customer drop locations,
- visit sequence,
- vehicle type,
- route distance,
- batch ETA,
- historical SLA,
- predicted late minutes, and
- simulated cost saving.

Example route:

```text
Store 2
   ↓
Store 3
   ↓
Store 1
   ↓
Drop 3
   ↓
Drop 1
   ↓
Drop 2
```

The solid arrows represent the selected route. Dashed arrows connect each store to its corresponding customer order.

To include the generated visualization in this README, place the image in the repository and use:

```markdown
![MILP-selected delivery route](MILP_Three_Order_Route_Clear.png)
```

---

## Repository Structure

```text
.
├── Corrected_Delivery_Optimization_Max_Batch_3.ipynb
├── Corrected_Delivery_Optimization_Max_Batch_3_executed.ipynb
├── amazon_delivery.csv
├── README.md
├── MILP_Three_Order_Route_Clear.png
└── delivery_outputs_max3/
    ├── delivery_time_model.joblib
    ├── model_comparison.csv
    ├── test_predictions.csv
    ├── stage3_order_sla_risk.csv
    ├── stage3_risk_summary.csv
    ├── stage4_all_candidates.csv
    ├── stage4_selected_candidates.csv
    ├── stage4_order_assignments.csv
    └── stage4_policy_summary.csv
```

---

## Installation

### Python dependencies

```bash
pip install numpy pandas matplotlib scikit-learn scipy joblib graphviz
```

Graphviz also requires a system installation.

### macOS

```bash
brew install graphviz
```

### Ubuntu or Google Colab

```bash
sudo apt-get install graphviz
```

### Windows

Install Graphviz and add its `bin` directory to the system `PATH`.

---

## Running the Project

1. Place the delivery CSV in the project directory.
2. Open the clean notebook:

```text
Corrected_Delivery_Optimization_Max_Batch_3.ipynb
```

3. Run all notebook cells.
4. Generated files will be stored under:

```text
delivery_outputs_max3/
```

To execute from the command line:

```bash
jupyter nbconvert \
  --to notebook \
  --execute Corrected_Delivery_Optimization_Max_Batch_3.ipynb \
  --output Corrected_Delivery_Optimization_Max_Batch_3_executed.ipynb \
  --ExecutePreprocessor.timeout=900
```

---

## Key Output Files

### `model_comparison.csv`

Performance of all delivery-time prediction models.

### `test_predictions.csv`

Order-level holdout predictions and absolute errors.

### `stage3_order_sla_risk.csv`

Order-level historical SLA, risk class, predicted late minutes, and evaluation fields.

### `stage3_risk_summary.csv`

Risk-class counts and actual late-delivery rates.

### `stage4_all_candidates.csv`

All feasible individual, two-order, and three-order candidate trips.

### `stage4_selected_candidates.csv`

Trips selected by the MILP.

### `stage4_order_assignments.csv`

Final order-to-trip and vehicle assignments.

### `stage4_policy_summary.csv`

Baseline versus optimized trips, distance, and simulated cost.

---

## Limitations

1. **Historical SLA is not an official Amazon SLA.**  
   It is estimated from the 75th percentile of similar training orders.

2. **Cost values are simulated.**  
   Actual labour, fuel, maintenance, contractor, and fleet costs are unavailable.

3. **Vehicle-specific predictions are predictive, not necessarily causal.**  
   Changing the vehicle feature produces a model-based counterfactual but does not prove causal impact.

4. **Preparation delay is used as a predictor.**  
   Therefore, the ETA is most appropriate at or after pickup information becomes available.

5. **Rider availability is not modelled.**  
   The MILP selects trips but does not assign named riders or enforce shift and scheduling constraints.

6. **The model batches at most three orders.**  
   Larger van batches would require more advanced route-generation methods such as OR-Tools, column generation, or large-neighbourhood search.

7. **Coordinates use a dataset-specific sign correction.**  
   Coordinate values are converted to absolute values because the public data contains inconsistent longitude signs.

8. **KNN categorical imputation uses ordinal code space.**  
   This introduces an approximate numerical geometry between categories.

---

## Future Improvements

- probabilistic or quantile ETA prediction,
- confidence intervals for delivery time,
- official promised-time or SLA data,
- real fleet and labour cost calibration,
- rider availability and shift constraints,
- rider starting-location modelling,
- dynamic traffic and weather feeds,
- complete vehicle-routing formulation,
- batching of four to six orders,
- OR-Tools route optimization,
- column generation for large candidate sets,
- fairness and workload-balancing constraints, and
- online re-optimization when new orders arrive.

---

## Disclaimer

This project is an independent academic and portfolio project based on a public dataset. It is not an official Amazon system, and the simulated operational outcomes should not be interpreted as measured Amazon business results.
