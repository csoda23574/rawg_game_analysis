# RAWG Game Analysis

This repository contains the analysis and modeling of RAWG game data to predict game "hits" using an XGBoost classification model and to predict game ratings using an XGBoost regression model.

## Project Overview

The goal of this project is to build predictive models for game success based on various game attributes from the RAWG database.

## Data

The analysis uses the `rawg_games_data.csv` dataset.

## Methodology

1.  **Data Loading and Preprocessing**:
    *   Loading the data using pandas.
    *   Handling missing values by dropping irrelevant columns (`metacritic`, `publishers`, `developers`, `description`) and rows with missing `name` or `released` dates.
    *   Converting the `released` column to datetime objects.
    *   Applying MultiLabelBinarizer for `genres` and `platforms` with frequency filtering.
    *   Applying MultiLabelBinarizer for `tags` by selecting the top N frequent tags.
    *   Creating interaction terms between `playtime` and one-hot encoded `genres`.
    *   Scaling numerical features (`rating`, `playtime`) using StandardScaler.

2.  **Target Variable Creation**:
    *   Defining a binary target variable 'Hit' based on scaled `rating` and interaction terms.
    *   Balancing the dataset for the 'Hit' classification task using undersampling.

3.  **Model Training (Classification)**:
    *   Splitting the balanced data into training and test sets.
    *   Training an XGBoost classification model to predict the 'Hit' status.
    *   Evaluating the classification model using classification report, confusion matrix, and ROC curve/AUC.
    *   Performing K-Fold Cross-Validation to assess model performance robustness.

4.  **Model Training (Regression)**:
    *   Defining the scaled 'rating' as the target variable for regression.
    *   Using the same processed features (excluding 'Hit', 'id', 'name', 'released', and original 'rating') for the regression model.
    *   Splitting the data into training and test sets for regression.
    *   (Note: The current notebook focuses on the classification model training and evaluation. The regression model training would follow a similar process using an appropriate regression objective in XGBoost and relevant regression evaluation metrics like Mean Squared Error or R-squared.)

## How to Run

1.  Clone the repository: `git clone [repository_url]`
2.  Open the `rawg_model_XGBoost_Reg_.ipynb` notebook in Google Colab or a Jupyter environment.
3.  Ensure you have the `rawg_games_data.csv` file available and update the data loading path if necessary.
4.  Run the cells sequentially.

## Files

*   `.gitignore`: Specifies files and directories to be ignored by Git.
*   `README.md`: Provides an overview and description of the project.
*   `rawg_model_XGBoost_Reg_.ipynb`: The Colab notebook containing the data processing, feature engineering, and XGBoost model implementation for classification and preparation for regression.
