# Medical Diagnosis Using Bayesian Network: Thyroid Disease Dataset

## Project Overview

This project focuses on **Exercise 4.3: Case of Medical Diagnosis** using the **Thyroid Disease Dataset** from the UCI Machine Learning Repository. The goal is to build and train a **Bayesian Network** classifier to diagnose thyroid diseases based on medical features and achieve an expected accuracy of at least 85%.

The dataset consists of features like **T3**, **T4**, **TSH** (Thyroid Stimulating Hormone), and the presence of **Goiter**. The labels classify the diagnosis into **Hyperthyroid**, **Hypothyroid**, and **Normal Thyroid Function**.

## Dataset Description

Dataset URL: [Thyroid Disease Dataset](https://archive.ics.uci.edu/dataset/102/thyroid+disease)

The dataset includes:
- **Features**:
  - T3
  - T4
  - TSH (Thyroid Stimulating Hormone)
  - Presence of Goiter (binary: 0 for absence, 1 for presence)
- **Labels**:
  - Hyperthyroid
  - Hypothyroid
  - Normal Thyroid Function

This dataset is a small and curated medical dataset consisting of 215 samples.

## Problem Statement

The objective is to:
- **Build a Bayesian Network classifier** using features like T3, T4, TSH, and the presence of Goiter.
- **Predict the diagnosis**: Hyperthyroid, Hypothyroid, or Normal Thyroid function based on the given features.
- **Achieve an accuracy of ≥ 85%** using the Bayesian approach for medical diagnosis.

## Methodology

1. **Preprocessing**: 
   - The dataset was loaded and cleaned by removing missing values.
   - Features were scaled, and categorical variables were encoded using the `LabelEncoder`.

2. **Bayesian Network**:
   - A **Bayesian Network** was built using the features as independent variables influencing the diagnosis.
   - The network assumes that each feature (T3, T4, TSH, and Goiter presence) has a direct influence on the diagnosis.
   - The model was trained using the **Bayesian Estimator** from the `pgmpy` library, which estimates the parameters using the data and prior knowledge.

3. **Model Training and Testing**:
   - The dataset was split into 80% training and 20% testing sets using `train_test_split`.
   - A **Variable Elimination** algorithm was used for inference and prediction based on the trained Bayesian Network.

4. **Performance Evaluation**:
   - The accuracy of the Bayesian Network was calculated by comparing predicted labels with the true labels on the test set.

## Libraries Used

- **Pandas**: For data manipulation and preprocessing.
- **Numpy**: For numerical operations.
- **scikit-learn**: For data splitting and encoding.
- **pgmpy**: For building and training the Bayesian Network.
- **BayesianEstimator**: To fit the Bayesian Network model.
- **VariableElimination**: For making predictions using the Bayesian Network.

### Installation

To install all required libraries in your environment, run:

```bash
!pip install pandas numpy scikit-learn pgmpy
```
## Results

The Bayesian Network was trained on the thyroid dataset and achieved an accuracy >85% accuracy goal. This demonstrates the effectiveness of the Bayesian Network for medical diagnosis in this context.

## Future Improvements

1. **Feature Engineering**: Explore additional feature transformations to improve prediction accuracy.
2. **Hyperparameter Tuning**: Fine-tune the Bayesian Estimator's parameters for better performance.
3. **More Data**: Include more data or implement resampling techniques to handle any class imbalances for a more robust model.

## Usage

To run the model and make predictions on thyroid disease diagnosis:
1. Run the Python script to preprocess the data, train the model, and evaluate its performance.
2. After training, you can input patient features manually to get a diagnosis using the prediction function:

```python
def get_input_and_predict():
    T3 = float(input("T3: "))
    T4 = float(input("T4: "))
    TSH = float(input("TSH: "))
    Goitre = int(input("Goitre (0 for absence, 1 for presence): "))
    
    prediction = predict(T3, T4, TSH, Goitre)
    print(f"Predicted diagnosis: {le.inverse_transform([prediction])[0]}")
```

This will return the predicted thyroid function based on the given input.
