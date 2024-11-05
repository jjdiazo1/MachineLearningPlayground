# MachineLearningPlayground

Basic Machine Learning Models for Toy Datasets

## Project Overview

This project was developed as part of the **Fundamentals of Machine Learning** course in the Biomedical Engineering program, which I took as an optional class in my CS major. The primary goal is to apply and understand basic machine learning algorithms on toy datasets, allowing for a hands-on approach to fundamental concepts. The project explores techniques such as linear regression, polynomial regression, logistic regression, simple neural networks, and clustering using the K-means algorithm.

## Showcase

### Greenhouse Analysis
---
<img width="944" alt="Screenshot 2024-11-03 at 4 24 28 PM" src="https://github.com/user-attachments/assets/a14b68d8-0f28-4841-a8e1-d28b2fe9e0ef">

<img width="944" alt="Screenshot 2024-11-03 at 4 25 41 PM" src="https://github.com/user-attachments/assets/d9a2a255-8869-4ddf-a7ce-e0f0c2e5a690">


### ToyAnalysis
---
<img width="944" alt="Screenshot 2024-11-03 at 4 43 24 PM" src="https://github.com/user-attachments/assets/a2154e5d-ed12-4e2a-8d42-292f4f925192">

### Logistic Regression
---
<img width="1437" alt="Screenshot 2024-11-03 at 4 59 04 PM" src="https://github.com/user-attachments/assets/deeb62de-8fad-4efd-94f1-9328d9aae98e">

### KMeans
---
<img width="942" alt="Screenshot 2024-11-03 at 4 54 04 PM" src="https://github.com/user-attachments/assets/0a8e2ebc-6cc2-46a5-be3c-905b97501751">

## Project Structure

This repository contains multiple Python scripts, each focusing on a specific algorithm or analysis. The datasets used in this project are simplified or "toy" datasets, provided for academic purposes.

### Files and Analysis

- **Linear and Polynomial Regression** (`toy_analysis.py`)
  - **Description**: This script explores different regression models (linear, quadratic, cubic, fourth, and fifth degree) to fit toy data in `datosToy.txt`.
  - **Key Steps**:
    - Scatter plot of the data.
    - Estimation of parameters for each regression model using least squares.
    - Visualization of fitted lines/polynomials and analysis of model fit.
    
- **Logistic Regression** (`regresion_logistica.py`)
  - **Description**: A logistic regression model is applied to classify 2D data points in `datosNN.txt`.
  - **Key Steps**:
    - Define weights and bias.
    - Apply the sigmoid function and classify points based on a threshold.
    - Plot the decision boundary and scatter plot of classified data points.

- **Basic Neural Network** (`neural_network.py`)
  - **Description**: Implements a simple feedforward neural network with a specified architecture for binary classification, as described in the provided academic exercise.
  - **Key Steps**:
    - Define individual neurons with activation functions (ReLU and Sigmoid).
    - Propagate inputs through the network.
    - Classify outputs and visualize the decision boundary.

- **K-means Clustering** (`k_means.py`)
  - **Description**: Uses the K-means algorithm to cluster patient locations (coordinates) in `coordenadasPacientes.txt`, aiming to optimize healthcare center placement.
  - **Key Steps**:
    - Perform initial scatter plot of patient locations.
    - Apply K-means clustering with 4 clusters.
    - Visualize clusters and compare initialization methods (`k-means++` and `random`).

- **Greenhouse Temperature and Humidity Analysis** (`greenhouse_analysis.py`)
  - **Description**: Analyzes the relationship between temperature and relative humidity in a greenhouse dataset (`greenhouse.txt`). Detects anomalies in new temperature-humidity readings.
  - **Key Steps**:
    - Scatter plot of temperature vs. humidity.
    - Linear regression model to fit temperature-humidity relationship.
    - Define a rule for anomaly detection based on residuals.
    - Visualize typical vs. anomalous data points in new measurements.

## How to Run

1. **Install dependencies**: This project uses only Python’s standard libraries (`numpy`, `matplotlib`). Install these if not already available.

   ```bash
   pip install numpy matplotlib
   ```

2. **Run scripts**: Each script can be executed independently. Run:

    ```bash
     python script_name.py
    ```

