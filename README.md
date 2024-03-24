# Principal Component Analysis (PCA) for Dimension Reduction

## Overview
This repository contains code for performing Principal Component Analysis (PCA) for dimensionality reduction. PCA is a popular technique used in machine learning and data analysis to reduce the number of dimensions in a dataset while preserving most of its variance. 

In this implementation, we have a Python class named `PCA` defined in `pca.py`, which provides methods for performing PCA. The PCA class is utilized in `main.py` to demonstrate how to use PCA for dimensionality reduction.

## Requirements
To run this code, you need to have the following dependencies installed:
- NumPy
- scikit-learn

You can install these dependencies using pip:
```
pip install numpy scikit-learn
```

## Usage
1. Clone this repository to your local machine or download the files `pca.py` and `main.py`.
2. Navigate to the directory where the files are located.
3. Run the following command in your terminal or command prompt:
   ```
   python main.py
   ```

## Files
- `pca.py`: Contains the implementation of the PCA class for performing dimensionality reduction using Principal Component Analysis.
- `main.py`: Demonstrates how to use the PCA class to perform dimensionality reduction on the Iris dataset from scikit-learn.

## Running the Code
Upon running `main.py`, the program will load the Iris dataset from scikit-learn, apply PCA for dimensionality reduction.

## Note
The Iris dataset is used here for demonstration purposes. You can replace it with your own dataset by modifying the `main.py` file accordingly.