# Derby — MLB Home Run Predictor

Derby is a machine-learning powered MLB analytics platform designed to predict the probability of individual players hitting a home run in a given matchup.

The project combines **MLB player and pitcher statistics, feature engineering, predictive modeling, and automated data updates** to generate daily home run predictions and rank the best candidates.

## Project Overview

Derby analyzes historical and recent MLB performance data to estimate a player's probability of hitting a home run.

The system processes player and pitcher statistics, creates predictive features, and feeds them into an **XGBoost classification model**. The resulting probabilities are used to rank players and generate daily home run predictions.

### Key Goals

* Predict individual player home run probabilities
* Analyze hitter and pitcher performance
* Automatically update MLB statistics
* Rank players based on predicted HR probability
* Provide an accessible platform for viewing daily predictions
* Apply machine learning to real-world sports data

##  Machine Learning Model

Derby uses an **XGBoost classification model** to predict whether a player will hit a home run.

The model is trained using historical MLB data and incorporates information from both hitters and opposing pitchers.

### Model Configuration

* **Algorithm:** XGBoost Classifier
* **Number of estimators:** 500
* **Maximum depth:** 6
* **Learning rate:** 0.03
* **Subsample:** 0.90
* **Column sampling:** 0.90
* **Probability calibration:** CalibratedClassifierCV

Calibration is used to improve the reliability of the predicted probabilities rather than simply producing binary predictions.

## Dataset

The project combines multiple sources of MLB performance data, including player and pitcher statistics.

The current modeling dataset contains approximately:

* **104,500+ observations**
* **24 predictive features**
* **10.6% home run label rate**
* **83,600+ training observations**
* **20,900+ test observations**

Features are designed to capture factors that can influence home run probability, including hitter performance, pitcher performance, and recent statistical trends.

## Model Performance

The model is evaluated using several metrics rather than relying solely on prediction accuracy.

Current test-set results include:

| Metric                        | Result |
| ----------------------------- | -----: |
| AUC                           |  0.609 |
| Log Loss                      |  0.325 |
| Brier Score                   |  0.091 |
| Average Predicted Probability |  0.103 |

These metrics help evaluate both the model's ability to distinguish between outcomes and the quality of its probability estimates.

## Data Pipeline

Derby includes scripts for processing and updating MLB data.

The general workflow is:

**MLB Data → Data Cleaning → Feature Engineering → Model → Probability Calibration → Player Rankings → Daily Predictions**

Key project files include:

```text
mlb-hr-predictor/
│
├── data.csv
├── player_stats.csv
├── pitcher_stats.csv
├── raw_players.csv
├── raw_pitchers.csv
│
├── mlb_data.py
├── refresh_live_stats.py
│
├── model files
├── application files
└── README.md
```

## Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Machine Learning**
* **Statistical Analysis**
* **Data Engineering**
* **Git & GitHub**
* **Web Development**

## Future Improvements

Derby is an ongoing project. Planned improvements include incorporating additional contextual variables such as:

* Ballpark dimensions
* Batter spray/pull tendencies
* Left-field/right-field asymmetry
* Recent barrel-rate splits
* Recent home-run-rate splits
* Additional pitcher/hitter matchup data
* Improved model calibration
* Expanded historical datasets

The goal is to continually improve both the predictive performance and usefulness of the platform.

## Project Purpose

Derby was created as a personal sports analytics project to apply machine learning and data science techniques to a real-world prediction problem.

Rather than relying on simple statistical rankings, the project attempts to combine multiple variables into a probabilistic model that can continuously generate and evaluate MLB home run predictions.

---

**Author:** [Caden Ryker]
**GitHub:** https://github.com/royalss-code/mlb-hr-predictor


To start go to file on desktop and type cmd into file directory

To update live stats daily use 

python refresh_live_stats.py
Then push player_stats.csv and pitcher_stats.csv into GitHub Repo

 For retraining model and Building Data use

 python build_real_dataset.py
 python train_model.py
 python test_model.py

 For running the app locally

 python app.py
 

