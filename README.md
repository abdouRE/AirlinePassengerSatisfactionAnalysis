# ✈️ Airline Passenger Satisfaction — Unsupervised Segmentation Pipeline

> An end-to-end K-Means clustering pipeline that discovers passenger behavioral personas from raw survey data — without ever seeing the satisfaction label — and validates the discovered structure against ground-truth outcomes.

---

## 📌 Overview

Airlines collect rich behavioral survey data from millions of passengers, but raw satisfaction scores alone don't explain *who* is satisfied or *why*. This project applies an unsupervised segmentation approach to a large-scale airline survey dataset (~130,000 passengers) to:

- Discover distinct passenger personas purely from behavioral and service-rating signals
- Validate that the discovered structure aligns with real satisfaction outcomes
- Identify the key in-cabin service drivers that separate satisfied from dissatisfied travelers
- Deliver actionable, data-backed recommendations for airline service investment

The satisfaction label is **never used as a clustering input** — it is introduced only at the validation stage, ensuring that segments emerge organically rather than by design.

---

## 📁 Dataset

| Property | Detail |
|---|---|
| **Source** | [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) |
| **Size** | ~130,000 passengers (train + test combined) |
| **Features** | 22 variables: passenger demographics, flight details, and 14 in-flight service ratings (1–5 scale) |
| **Target** | `satisfaction` — used **only** for post-hoc validation |

---

## 🔁 Pipeline Stages

```
1. Data Loading & Combination
2. EDA — Missing Values
3. EDA — Outlier Detection & Treatment (log1p)
4. Feature Selection, Encoding & Scaling
5. Optimal k Selection (Elbow + Silhouette)
6. Final K-Means Clustering (k=2)
7. Dimensionality Reduction & Visualization (PCA)
8. Cluster Validation vs. Ground-Truth Labels
9. Cluster Profiling — Standardized Centroid Heatmap
```

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| `pandas` | Data loading, wrangling, profiling |
| `numpy` | Numerical transforms (log1p) |
| `scikit-learn` | KMeans, PCA, StandardScaler, silhouette_score |
| `matplotlib` | Elbow curve, PCA scatter, bar charts |
| `seaborn` | Boxplots, heatmaps, styled visualizations |

---

## ⚙️ Preprocessing Decisions

### Missing Values
`Arrival Delay in Minutes` contains a small number of nulls. Given the feature's heavy right-skew, missing values are imputed with the **median** — which is robust to the extreme high values characteristic of delay distributions.

### Outlier Treatment — Log1p Transformation
`Flight Distance`, `Departure Delay in Minutes`, and `Arrival Delay in Minutes` are heavily right-skewed. Rather than capping at arbitrary thresholds, `log1p(x) = log(1 + x)` is applied because it:
- Compresses extreme values without discarding them
- Preserves zero (`log1p(0) = 0`) and relative ordering
- Is theoretically sound for distance-based algorithms like K-Means, where outliers disproportionately distort centroid estimates

### Feature Exclusions
The following columns are dropped before building the feature matrix:

| Column | Reason |
|---|---|
| `Unnamed: 0`, `id` | Row identifiers — no behavioral signal |
| `Gender` | No meaningful distinction for service segmentation |
| `satisfaction` | Validation label — excluded to prevent leakage |

### Encoding & Scaling
Categorical variables (`Customer Type`, `Type of Travel`, `Class`) are one-hot encoded with `drop_first=True`. All features are then standardized with `StandardScaler` to give each dimension equal weight in the Euclidean distance calculation used by K-Means.

---

## 📊 Key Results

### Optimal k

Both the **Elbow Method** (WCSS inflection) and the **Silhouette Score** peak consistently indicate **k=2**, naturally matching the dataset's binary satisfaction structure — without that label ever being consulted.

### Discovered Segments

| Cluster | Label | Size | Profile |
|---|---|---|---|
| **Cluster 1** | Satisfied Business Traveler | ~52% | Business-class, corporate travel, high ratings on entertainment, cleanliness, comfort, catering, and online boarding |
| **Cluster 0** | Dissatisfied Leisure Economy Passenger | ~48% | Economy-class, personal travel, below-average ratings across all in-cabin service dimensions |

### Validation
The organic clustering achieves strong alignment with ground-truth `satisfaction` labels, confirming that the discovered structure reflects genuine behavioral patterns rather than algorithmic artifacts.

### A Critical Non-Finding
Delay metrics (`Departure Delay in Minutes`, `Arrival Delay in Minutes`) and `Gate location` all sit within ±0.07 SD of the global mean across both clusters. This means **schedule reliability is not what separates satisfied from dissatisfied passengers** — the satisfaction gap is driven entirely by in-cabin experience quality.

---

## 💡 Business Recommendations

**For the Leisure Economy segment (Cluster 0):**
The root cause of dissatisfaction is a gap between passenger expectations and the economy cabin experience. Investment should prioritize:
1. Inflight entertainment quality and availability
2. Seat comfort upgrades in economy
3. Cabin cleanliness standards
4. Food and beverage offering
5. Online boarding experience

Chasing delay reduction alone will **not** close the satisfaction gap for this segment.

**For the Business segment (Cluster 1):**
Loyalty is maintained by protecting premium in-cabin services. Degradation in entertainment, comfort, or cleanliness carries disproportionate risk for this group.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Data
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) and place `train.csv` and `test.csv` in the project root.

### Run
Open the notebook in Jupyter, Kaggle, or Google Colab and execute all cells in order.

```bash
jupyter notebook Airline_Passenger_Satisfaction.ipynb
```

---

## 📂 Project Structure

```
├── Airline_Passenger_Satisfaction.ipynb          # Main pipeline notebook
├── train.csv                                        # Training split (from Kaggle)
├── test.csv                                         # Test split (from Kaggle)
└── README.md
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).
