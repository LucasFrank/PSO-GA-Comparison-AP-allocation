# Intelligent Resource Allocation in Wireless Networks

Machine Learning project for **wireless network traffic prediction and intelligent Access Point (AP) allocation**, using real-world network data and meta-heuristic hyperparameter optimization.

> **Key Results:** The optimized **PSO-MLP** model achieved **R² = 0.9434**, outperforming the evaluated Decision Tree, LSTM, and GRU models. When applied to network resource management, the prediction mechanism achieved **95.18% accuracy in determining the number of Access Points required**, while maintaining the defined network SLA.

## Project Summary

This project investigates how **Machine Learning and Computational Intelligence** can improve resource allocation in large-scale wireless networks.

The main objective is to predict the number of users connected to a wireless network and use this prediction to determine how many **Access Points (APs)** need to remain active at a given time.

Accurately predicting network demand makes it possible to avoid keeping unnecessary APs active during periods of low utilization while maintaining sufficient network capacity during periods of high demand.

The project combines:

- Exploratory Data Analysis (EDA)
- Time-series feature engineering
- Multilayer Perceptron (MLP)
- Decision Tree (DT)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Particle Swarm Optimization (PSO)
- Genetic Algorithm (GA)
- Hyperparameter optimization
- Wireless network traffic prediction
- Intelligent resource allocation

The experiments were conducted using a real university wireless network dataset containing more than **14 million connection records**, **36,000 unique devices**, and **108 Access Points** collected over four months.

### Highlights

- **Problem:** Wireless network user-load prediction and Access Point allocation
- **Dataset:** Real-world university wireless network logs
- **Original Records:** 14,295,222
- **Unique Devices:** ~36,000
- **Access Points:** 108
- **Collection Period:** 4 months
- **Models Evaluated:** MLP, Decision Tree, LSTM, GRU
- **Optimized Models:** PSO-MLP, GA-MLP, PSO-DT, GA-DT
- **Optimization Algorithms:** Particle Swarm Optimization and Genetic Algorithm
- **Best Prediction Model:** PSO-MLP
- **Best R²:** 0.9434
- **MAE:** 0.0214
- **RMSE:** 0.0363
- **AP Allocation Accuracy:** 95.18%
- **Technologies:** Python, Jupyter Notebook, Pandas, Matplotlib, Seaborn, Machine Learning

---

## Problem

Large wireless networks must continuously provide sufficient resources to users while avoiding unnecessary resource consumption.

The number of connected users varies significantly throughout the day, creating periods of:

- High network demand
- Low network utilization
- Traffic peaks
- Idle resources

Keeping every Access Point active at all times can result in unnecessary resource and energy consumption.

On the other hand, disabling too many APs may reduce available bandwidth and negatively affect:

- Quality of Service (QoS)
- Quality of Experience (QoE)
- Service Level Agreement (SLA)

The goal of this project is therefore to:

> **Predict the number of connected users and dynamically estimate the number of Access Points required to support the predicted network demand.**

---

## Proposed Approach

The project follows two main stages:

1. **Predict the number of users connected to the wireless network**
2. **Determine the number of Access Points required to support those users**

The overall workflow can be represented as:

```text
Wireless Network Logs
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Temporal Feature Engineering
        │
        ▼
Machine Learning Models
        │
        ├── MLP
        └── Decision Tree
        │
        ▼
Hyperparameter Optimization
        │
        ├── Particle Swarm Optimization
        └── Genetic Algorithm
        │
        ▼
User Load Prediction
        │
        ▼
Access Point Allocation
        │
        ▼
Network Resource Optimization
```

Four optimized prediction approaches were evaluated:

```text
PSO + MLP  → PSO-MLP
GA  + MLP  → GA-MLP

PSO + DT   → PSO-DT
GA  + DT   → GA-DT
```

The optimized models were compared using multiple regression metrics.

---

## Dataset

The dataset was collected from a **large university campus wireless network in Brazil**.

Data were gathered during a period of **four months** from **108 unique Access Points**.

Approximately **36,000 unique devices** generated about **14 million network access requests** during the collection period.

### Dataset Scale

| Attribute | Value |
|---|---:|
| Collection Period | 4 months |
| Access Points | 108 |
| Unique Devices | ~36,000 |
| Original Records | 14,295,222 |
| Attributes | 4 |

Each network record contains:

| Feature | Description |
|---|---|
| `time of day` | Time when the host connected to an AP |
| `connection time` | Duration of the host connection |
| `host` | Host/device identification |
| `access point` | Access Point identification |

---

## Data Preprocessing

The original dataset contained:

```text
14,295,222 records
```

No missing values were identified.

However, the connection duration ranged from extremely short connections to connections lasting approximately 100 days.

Connections lasting less than **30 seconds** were removed because they were considered unlikely to represent meaningful network traffic.

After preprocessing, the dataset contained:

```text
7,044,341 records
```

Therefore, approximately half of the original records remained after filtering short-lived connections.

This preprocessing step helped focus the analysis on connections more likely to represent actual network utilization.

---

## Exploratory Data Analysis

An **Exploratory Data Analysis (EDA)** was performed to understand user behavior and wireless network usage patterns.

The analysis was implemented using:

- Jupyter Notebook
- Python 3.8.5
- Pandas
- Matplotlib
- Seaborn

The analysis investigated:

- Network usage throughout the day
- Peak connection periods
- User mobility patterns
- Access Point utilization
- Connection duration
- Differences between campus locations

### Network Usage Patterns

The data showed that most network activity occurred between:

```text
09:00 AM – 06:00 PM
```

The highest traffic peak occurred around lunchtime:

```text
11:00 AM – 01:00 PM
```

The analysis also revealed that approximately **6,000 users** could remain connected for around **5 minutes** in frequently accessed campus regions.

Different Access Points presented different usage patterns depending on their physical location.

APs near:

- Classrooms
- Restaurants
- Cafés
- Libraries

showed stronger traffic peaks.

Other APs located primarily along transit paths presented more stable traffic behavior.

---

## Time-Series Features

The prediction model uses both **recent observations and historical observations**.

The input can be represented as:

```text
Recent observations:
V(t), V(t-1), ..., V(t-P)

Historical observations:
same time on previous days
```

Where:

- `P` represents recent observations
- `Q` represents historical observations from previous days
- Each recent observation is aggregated using **5-minute intervals**

The model predicts:

```text
V(t+1)
```

which represents the number of connected users in the next time interval.

This approach allows the model to learn:

- Short-term network behavior
- Daily patterns
- Historical patterns
- Network seasonality

---

## Training Strategy

The dataset was divided into training, validation, and testing data.

```text
80% → Training + Validation
20% → Testing
```

Within the 80% training/validation portion:

```text
2/3 → Training
1/3 → Validation
```

The optimization algorithms use the training and validation data to search for improved model hyperparameters.

The final optimized models are then evaluated using the test dataset.

---

## Machine Learning Models

Four base Machine Learning architectures were initially evaluated:

- Multilayer Perceptron (MLP)
- Decision Tree (DT)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

### Baseline Performance

Before hyperparameter optimization:

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| **MLP** | **0.9331** | **0.0236** | **0.0397** |
| DT | 0.8905 | 0.0281 | 0.0508 |
| LSTM | 0.9083 | 0.0282 | 0.0465 |
| GRU | 0.9136 | 0.0268 | 0.0451 |

The **MLP achieved the highest R² and lowest errors among the evaluated baseline models**.

Because MLP and Decision Tree provided strong results while avoiding the additional training cost of recurrent neural networks, they were selected for hyperparameter optimization.

---

## Multilayer Perceptron

The MLP architecture consists of:

```text
Input Layer
     │
     ▼
Hidden Layer
     │
     ▼
Output Layer
     │
     ▼
Predicted Number of Users
```

The network contains a single hidden layer.

The output layer contains one neuron representing the predicted number of connected users.

### Optimized MLP Hyperparameters

The optimization algorithms search for:

- Number of neurons
- Learning rate

---

## Decision Tree

A Decision Tree regression model was also evaluated.

The optimized hyperparameters were:

- Maximum depth (`MD`)
- Minimum samples split (`MSS`)

The regression criterion used by the model was **Mean Squared Error (MSE)**.

---

## Hyperparameter Optimization

Two meta-heuristic optimization algorithms were evaluated:

### Particle Swarm Optimization

Particle Swarm Optimization (PSO) represents candidate solutions as particles moving through the hyperparameter search space.

Each particle stores:

```text
Current Position
Personal Best (pBest)
Global Best (gBest)
```

The algorithm repeatedly updates particle positions and evaluates the corresponding Machine Learning model.

### Genetic Algorithm

The Genetic Algorithm represents model hyperparameters as genes.

The optimization process applies:

- Population initialization
- Fitness evaluation
- Selection
- Crossover
- Mutation

The mutation rate used in the proposed algorithm was:

```text
10%
```

The process searches for combinations of hyperparameters that improve model performance.

---

## Optimized Models

The combination of Machine Learning models and optimization algorithms resulted in four models:

| Model | Machine Learning | Optimization |
|---|---|---|
| PSO-MLP | Multilayer Perceptron | Particle Swarm Optimization |
| GA-MLP | Multilayer Perceptron | Genetic Algorithm |
| PSO-DT | Decision Tree | Particle Swarm Optimization |
| GA-DT | Decision Tree | Genetic Algorithm |

---

## Evaluation Metrics

The prediction models were evaluated using three regression metrics.

### Coefficient of Determination

**R²** measures how much of the variance in the dependent variable can be explained by the model.

```text
Higher R² → Better
```

### Mean Absolute Error

**MAE** measures the average absolute difference between predicted and observed values.

```text
Lower MAE → Better
```

### Root Mean Squared Error

**RMSE** measures the standard deviation of prediction errors.

```text
Lower RMSE → Better
```

For the Access Point allocation stage, **accuracy** was also used.

---

## Prediction Results

The optimized models achieved the following average results:

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| **PSO-MLP** | **0.9434** | **0.0214** | **0.0363** |
| GA-MLP | 0.9434 | 0.0215 | 0.0365 |
| PSO-DT | 0.9243 | 0.0233 | 0.0427 |
| GA-DT | 0.9234 | 0.0234 | 0.0422 |

The values represent averages obtained after repeated model training and prediction.

### Best Model

The **PSO-MLP** presented the best overall combination of metrics:

```text
R²   = 0.9434
MAE  = 0.0214
RMSE = 0.0363
```

The optimized MLP configuration used:

```text
Number of neurons: 129
Learning rate:      0.0008
```

The GA-MLP obtained the same average R² but slightly higher MAE and RMSE.

---

## Impact of Hyperparameter Optimization

The baseline MLP achieved:

```text
R²   = 0.9331
MAE  = 0.0236
RMSE = 0.0397
```

After PSO optimization:

```text
R²   = 0.9434
MAE  = 0.0214
RMSE = 0.0363
```

Therefore, the optimization process improved the model across all three evaluation metrics.

### MLP Comparison

| Metric | Default MLP | PSO-MLP |
|---|---:|---:|
| R² | 0.9331 | **0.9434** |
| MAE | 0.0236 | **0.0214** |
| RMSE | 0.0397 | **0.0363** |

Both PSO and GA demonstrated similar effectiveness when optimizing the MLP.

---

## Network Resource Management

After validating the prediction models, the best-performing model was applied to a practical **network resource management scenario**.

The objective was to determine how many Access Points need to remain active according to the predicted number of connected users.

The scenario assumes:

```text
Average user traffic:     10–15 Mbps
Ethernet link per AP:     1 Gbps
Maximum users per AP:     65
Available APs:            up to 7
Target SLA:               ~95%
```

The number of required Access Points can be estimated as:

```text
Required APs = round(Predicted Users / 65)
```

This makes it possible to deactivate unnecessary APs during periods of lower network demand.

---

## Access Point Allocation Results

The prediction model was used to estimate whether:

```text
1 AP
2 APs
3 APs
4 APs
```

were required at a given time.

### Predicted vs. Real Allocation

| Required APs | Predicted Distribution | Real Distribution |
|---:|---:|---:|
| 1 | 84.28% | 83.06% |
| 2 | 11.47% | 12.36% |
| 3 | 3.97% | 4.04% |
| 4 | 0.28% | 0.54% |

The predicted resource allocation closely followed the real network demand.

### Allocation Accuracy

The model correctly predicted:

| Number of APs | Accuracy |
|---:|---:|
| 1 AP | 98.72% |
| 2 APs | 78.17% |
| 3 APs | 80.60% |
| 4 APs | ~50% |

The overall allocation accuracy was:

```text
95.18%
```

When only under-allocation cases that could negatively affect QoS are considered harmful, the reported accuracy approaches:

```text
~97%
```

---

## Practical Impact

The prediction system demonstrates how Machine Learning can be used not only for forecasting but also to support **real network management decisions**.

By predicting the number of connected users, the network can estimate how many APs need to remain active.

This can help:

- Reduce unnecessary active network resources
- Reduce potential energy consumption
- Prevent network saturation
- Improve bandwidth utilization
- Maintain Quality of Service
- Maintain Quality of Experience
- Preserve network SLA
- Adapt resources according to network demand

Instead of keeping every AP active continuously:

```text
Predicted User Demand
        │
        ▼
Required Network Capacity
        │
        ▼
Required Number of APs
        │
        ▼
Activate Only Necessary Resources
```

---

## Technology Stack

### Programming & Analysis

- Python 3
- Python 3.8.5 for EDA
- Jupyter Notebook

### Data Science

- Pandas
- Matplotlib
- Seaborn
- Exploratory Data Analysis
- Data Preprocessing
- Time-Series Feature Engineering

### Machine Learning

- Multilayer Perceptron
- Decision Tree
- LSTM
- GRU
- Regression
- Time-Series Prediction

### Computational Intelligence

- Particle Swarm Optimization
- Genetic Algorithm
- Hyperparameter Optimization

### Evaluation

- R²
- MAE
- RMSE
- Accuracy
- Confusion Matrix

### Application Domain

- Wireless Networks
- Network Traffic Prediction
- Access Point Management
- Network Resource Allocation
- QoS / QoE
- Service Level Agreements

---

## Key Findings

The main findings of this work were:

- A real wireless network dataset with more than **14 million records** was analyzed.
- The dataset included approximately **36,000 unique devices** and **108 Access Points**.
- Exploratory Data Analysis revealed clear temporal and location-dependent network usage patterns.
- MLP outperformed Decision Tree, LSTM, and GRU in the initial comparison.
- PSO and GA successfully improved the hyperparameters of MLP and Decision Tree models.
- **PSO-MLP achieved R² = 0.9434, MAE = 0.0214, and RMSE = 0.0363.**
- Hyperparameter optimization improved all three MLP evaluation metrics compared with the default configuration.
- The prediction model was successfully applied to a network resource management scenario.
- The system achieved **95.18% accuracy when predicting the required number of Access Points**.
- The proposed approach demonstrates how Machine Learning predictions can be translated into practical infrastructure allocation decisions.

---

## Reproducibility

The project was designed with reproducibility in mind.

The research provides:

- Dataset
- Jupyter Notebook implementation
- Prediction models
- Optimization algorithms
- Experimental results

The implementation allows the experiments to be reproduced and extended with additional prediction and simulation components.

---

## Future Work

Several extensions were identified for future research.

### Long-Term Traffic Prediction

Develop models capable of predicting wireless network traffic further into the future.

This could allow network topology changes to be planned with greater anticipation.

### Automated Network Reconfiguration

Integrate the prediction system with network management frameworks capable of automatically applying resource allocation decisions.

Potential technologies include:

- Software-Defined Networking (SDN)
- Ryu Controller
- OpenWRT
- DD-WRT

This could evolve the current prediction pipeline into a closed-loop intelligent network management system:

```text
Network Data
     │
     ▼
Traffic Prediction
     │
     ▼
Resource Decision
     │
     ▼
SDN / AP Controller
     │
     ▼
Automatic Network Reconfiguration
```

---

---

## Academic Context

This repository contains the implementation and experimental resources associated with the research presented in:

**Intelligent Resource Allocation in Wireless Networks: Predictive Models for Efficient Access Point Management**

The project was originally developed as **PSO-GA-Comparison**, focusing on comparing **Particle Swarm Optimization (PSO)** and **Genetic Algorithm (GA)** for hyperparameter optimization of Machine Learning models applied to wireless network traffic prediction and resource allocation.

### Authors

- Lucas R. Frank
- Antonino Galletta
- Lorenzo Carnevale
- Alex B. Vieira
- Edelberto Franco Silva

### Institutions

**Federal University of Juiz de Fora (UFJF)**  
Department of Computer Science  
Juiz de Fora, Brazil

**University of Messina**  
Department of Mathematical and Computer Science, Physics and Earth Sciences  
Messina, Italy

---

## Citation

If you use this dataset, code, or results in your research, please cite:

> Frank, L. R., Galletta, A., Carnevale, L., Vieira, A. B., & Silva, E. F. (2024).  
> **Intelligent resource allocation in wireless networks: Predictive models for efficient access point management.**  
> *Computer Networks, 254*, 110762.

### BibTeX

```bibtex
@article{frank2024intelligent,
  title={Intelligent resource allocation in wireless networks: Predictive models for efficient access point management},
  author={Frank, Lucas R. and Galletta, Antonino and Carnevale, Lorenzo and Vieira, Alex B. and Silva, Edelberto Franco},
  journal={Computer Networks},
  volume={254},
  pages={110762},
  year={2024}
}
