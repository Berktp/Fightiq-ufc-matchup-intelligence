# Fightiq-ufc-matchup-intelligence
# FightIQ – Explainable UFC Matchup Intelligence
Webapp link:https://fightiq-ufc-matchup-intelligence-fpadn2dwne9pixkob5aams.streamlit.app/

## Overview
FightIQ is a machine learning project focused on analyzing UFC matchups using historical fight, fighter, and event data.

## Goal
The goal of this project is to build an explainable system that estimates pre-fight win probability and highlights the main factors behind each prediction.

## Core Datasets
- `Fights.csv`
- `Fighters Stats.csv`
- `Fighters.csv`
- `Events.csv`

## Planned Workflow
- data cleaning
- exploratory data analysis (EDA)
- hypothesis testing
- feature engineering
- model comparison
- explainability
- Streamlit web app

## Project Structure
- `data/` → raw, interim, and processed datasets
- `notebooks/` → analysis notebooks
- `src/` → reusable Python scripts
- `app/` → web application
- `reports/` → project notes and report
- `figures/` → visual outputs

## Exploratory Data Analysis and Hypothesis Testing

At this stage of the project, the main focus is on **data cleaning, exploratory data analysis (EDA), and hypothesis testing**. The goal is to understand the structure of the UFC dataset, identify meaningful patterns, and evaluate whether selected fight-related factors are statistically associated with winning outcomes.

### Data Preparation
The cleaned base table used in this stage was constructed by combining:
- `Fights.csv`
- `Fighters Stats.csv`
- `Fighters.csv`
- `Events.csv`

The cleaned analysis file is:

- `data/processed/model_base_table_clean_names.csv`

During preprocessing:
- duplicate rows were removed
- date values were standardized
- relevant fighter and event information was merged
- inconsistent raw fighter name columns were removed from the analysis version
- canonical fighter name columns were retained for consistency
- post-fight leakage columns such as finish details and in-fight statistics were excluded from the base analytical workflow

### EDA Objectives
The exploratory analysis aims to:
- summarize the overall structure of the dataset
- inspect missing values and variable types
- examine class balance in the target variable
- visualize distributions of key numerical and categorical variables
- analyze fight outcome patterns across weight classes, stance types, and time formats
- explore relationships between engineered matchup features and winning

### EDA Outputs
The EDA pipeline generates:
- dataset overview tables
- missing value summaries
- numeric and categorical summary statistics
- target distribution analysis
- yearly fight trends
- category-based frequency plots
- histograms of selected matchup features
- boxplots comparing feature distributions across fight outcomes
- correlation heatmaps for selected numerical variables

Generated outputs are stored in:
- `reports/eda/`
- `figures/eda/`

### Hypothesis Testing
To move beyond descriptive analysis, five hypothesis tests were conducted to examine whether selected pre-fight or matchup-related variables are associated with the probability of winning.

The following hypotheses were tested:

1. **Reach Advantage Hypothesis**  
   Fighters with a reach advantage are more likely to win.  
   **p-value: 0.371**

2. **Southpaw Advantage Hypothesis**  
   Southpaw fighters have a different win rate than orthodox fighters.  
   **p-value: 0.205**

3. **Grappling Advantage Hypothesis**  
   Fighters with stronger grappling-related metrics are more likely to win.  
   **p-value: 0.000...

4. **Experience Advantage Hypothesis**  
   More experienced fighters are more likely to win.  
   **p-value: 0.000...

5. **Striking Advantage Hypothesis**  
   Fighters with stronger striking-related metrics are more likely to win.  
   **p-value: 0.000...

### Statistical Approach
The hypothesis testing stage uses:
- contingency tables
- chi-square tests of independence
- p-value based decision making at **alpha = 0.05**

Decision rule:
- **p < 0.05** → reject the null hypothesis
- **p ≥ 0.05** → fail to reject the null hypothesis

### Summary of Current Findings
The current hypothesis testing results suggest that:
- **reach advantage** is not statistically significant in this dataset ❌
- **southpaw stance** does not show a statistically significant advantage ❌
- **grappling-related advantage** is significantly associated with winning ✅
- **experience advantage** is significantly associated with winning ✅
- **striking-related advantage** is significantly associated with winning ✅


These findings should be interpreted as **exploratory statistical results** at the current stage of the project.

### Important Methodological Note
Some fighter-level aggregate statistics included in the merged dataset may reflect **overall career summaries at the time of data collection**, rather than values strictly available before each historical fight. Because of this, these variables are acceptable for **EDA and exploratory hypothesis testing**, but they will be treated more carefully in the modeling stage to reduce potential temporal leakage.

## Machine Learning Modeling

After completing the data cleaning, EDA, and hypothesis testing stages, the project moved into the machine learning phase. The purpose of this stage was to evaluate how well matchup-level features can predict fight outcomes and to compare multiple modeling approaches under a realistic train-test setup.

### Modeling Objective
The modeling stage was designed to answer the following questions:
- Can fight outcomes be predicted at a useful level using historical UFC data?
- Which feature groups carry the strongest predictive signal?
- Which machine learning model performs best on the processed dataset?
- Can the resulting system remain interpretable while still achieving solid predictive performance?

### Feature Set Strategy
Two separate feature configurations were used during the modeling stage.

#### 1. Safe Feature Set
The safe feature set was designed to be methodologically cleaner and more conservative. It mainly includes:
- physical matchup features such as height, weight, and reach differences
- fighter-level height, weight, and reach values
- stance information
- weight class
- time format
- fight year
- missing-value indicator flags

This feature group was intended to reduce temporal leakage risk as much as possible.

#### 2. Enriched Feature Set
The enriched feature set extends the safe set by including broader performance-related aggregate features such as:
- `wins_diff`
- `losses_diff`
- `draws_diff`
- `avg_fight_time_diff`
- `kd_diff`
- `str_diff`
- `td_diff`
- `sub_diff`
- `ctrl_diff`
- `sig_str_pct_diff`
- `sub_att_diff`
- `ko_rate_diff`
- `sub_rate_diff`
- `dec_rate_diff`

This second configuration was used to test whether richer fighter-level statistical signals improve predictive performance.

### Models Trained
The following models were trained and compared:
- **Logistic Regression** as a baseline model
- **Random Forest** as a nonlinear ensemble model
- **XGBoost** as the strongest gradient-boosted tree benchmark

### Train-Test Split Strategy
A **temporal split** approach was used instead of a purely random split. Earlier fight years were used for training, while more recent fights were reserved for validation and test evaluation. This makes the evaluation more realistic and better aligned with a real prediction setting.

### Evaluation Metrics
The following metrics were used to evaluate model performance:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Brier Score
- Calibration Curve
- Confusion Matrix

These metrics were chosen to evaluate not only classification accuracy, but also ranking quality, probability quality, and class-balance behavior.

### Best Model
The best-performing model in the current pipeline was:
- **Model:** XGBoost
- **Feature Set:** Enriched
- **Validation ROC-AUC:** 0.7549
- **Test Accuracy:** 0.7091
- **Test Precision:** 0.7222
- **Test Recall:** 0.7674
- **Test F1-score:** 0.7441
- **Test ROC-AUC:** 0.7981
- **Test Brier Score:** 0.1792

### Interpretation of Modeling Results
The modeling results indicate that:
- the **safe feature set** alone produced relatively weak predictive performance
- the **enriched feature set** substantially improved model quality
- **XGBoost** delivered the best overall predictive performance
- **Random Forest** also performed strongly, especially in recall and F1-score
- **Logistic Regression** remained useful as a baseline and interpretability reference model

In practical terms, these results suggest that simple physical and contextual features alone are not sufficient to explain UFC outcomes at a strong level, while richer style- and performance-based features provide much stronger predictive signal.

### Calibration and Reliability
A calibration curve was generated for the best-performing model in order to evaluate whether predicted probabilities were reasonably aligned with observed outcomes. The resulting calibration pattern suggests that the probability estimates are usable and not purely overconfident, although there is still room for improvement in probability calibration.

### Confusion Matrix Interpretation
A confusion matrix was also generated for the best model. This helps show how well the model distinguishes wins and losses, and whether it tends to overpredict one class. In the current version, the model demonstrates a reasonable balance between identifying true wins and limiting false positives.

### Error Analysis
Error analysis was performed in order to better understand the cases where the model fails. Special attention was given to:
- highly confident wrong predictions
- possible upset fights
- style mismatches
- performance differences across fight contexts
- the possible effect of missing values, especially reach-related missingness

This analysis is important because it shows that the model is not only evaluated numerically, but also inspected qualitatively.

---

## Explainability

Because FightIQ is designed as an **explainable UFC matchup intelligence system**, interpretability is a core component of the project.

### Global Explainability
Global feature importance was extracted in order to identify which variables influence the model most strongly. The current importance outputs suggest that the strongest signals come from:
- finish-profile-related differences
- striking-related differences
- grappling/control differences
- experience-related features

### Local Explainability
In addition to global importance, the project also includes local matchup-level explanation logic. This means that for a selected matchup, the system attempts to explain:
- why Fighter A may be favored
- which side has the stronger striking profile
- which side has the stronger grappling profile
- where upset risk may come from

This layer is especially important for making the final application more readable and useful to non-technical users.

---

## Streamlit Web Application

To make the project more interactive and portfolio-ready, FightIQ also includes a **Streamlit-based web application**.

### App Purpose
The purpose of the app is to transform the notebook-based workflow into a more accessible and user-facing demo. Instead of only showing model results in notebooks, the app allows users to directly interact with fighters, matchup comparisons, and project insights.

### Current App Sections
The current app includes three main sections:

#### 1. Matchup Predictor
This section allows the user to:
- select two fighters
- choose a fight year
- choose a time format
- generate a predicted win probability
- inspect key advantages and risk flags
- compare fighters through striking and grappling visualizations
- view an overall fighter comparison table

#### 2. Fighter Explorer
This section allows the user to:
- inspect an individual fighter profile
- review physical attributes and career information
- examine striking metrics
- examine grappling metrics
- explore their presence in the cleaned historical dataset

#### 3. Project Insights
This section summarizes:
- model comparison outputs
- hypothesis testing results
- feature importance findings
- methodological notes

### App Design Goals
The Streamlit app was built to be:
- readable
- interactive
- explainable
- suitable for portfolio presentation
- aligned with the analytical logic of the project

The application currently functions as a **project demo** rather than a production-grade scouting or betting system.

---

## Limitations

The current version of the project has several limitations:
- some fighter-level aggregate variables may carry **temporal leakage risk**
- historical pre-fight information is not perfectly reconstructed for every fight
- some variables still contain missing values, especially reach-related features
- UFC fight outcomes are inherently noisy and influenced by many contextual factors not fully captured in the dataset
- the current Streamlit application is a research and portfolio demo rather than a production-grade deployment

These limitations are important and should be considered when interpreting both the statistical findings and the machine learning results.

---
## Author
Berk Talha Pala