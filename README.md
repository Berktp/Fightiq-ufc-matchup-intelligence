# Fightiq-ufc-matchup-intelligence
# FightIQ – Explainable UFC Matchup Intelligence

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

### Next Step
The next stage of the project will focus on:
- refining the modeling feature set
- reducing leakage risk
- applying machine learning methods
- comparing predictive models
- building an interpretable UFC matchup intelligence system
## Author
Berk Talha Pala