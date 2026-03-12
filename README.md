# ANN Classification and Regression

This repository contains two Artificial Neural Network (ANN) projects built with **PyTorch**:

- **ANN Classification** on the **Date Fruit dataset**
- **ANN Regression** on the **Power Plant dataset**

It also includes a **Streamlit app** for the regression model so the trained network can be tested through a simple web interface.

## Project Overview

The goal of this repository is to practice implementing feedforward neural networks for two common supervised learning tasks:

- **Classification**: predicting the class/category of a sample
- **Regression**: predicting a continuous numerical value

The repository demonstrates a full basic deep learning workflow:

1. Loading the dataset with Pandas
2. Splitting data into training and testing sets
3. Preprocessing with `StandardScaler`
4. Converting data into PyTorch tensors
5. Building ANN models using `torch.nn`
6. Training with Adam optimizer
7. Evaluating model performance
8. Visualizing loss curves
9. Deploying the regression model with Streamlit

## Repository Structure

```bash
ANN-classification/
│
├── ann_classification.ipynb   # ANN for multiclass classification
├── ann_reg.ipynb              # ANN for regression
└── app.py                     # Streamlit app for power plant regression
```

## 1) ANN Classification

The notebook `ann_classification.ipynb` implements a **multiclass classification model**.

### Dataset
- Dataset used: **DateFruit_Dataset.csv**
- Target column: **`Class`**
- Features: numerical fruit attributes such as area, perimeter, major axis, minor axis, eccentricity, color/statistical descriptors, etc.

### Preprocessing Steps
- The target labels are encoded using **LabelEncoder**
- Features are standardized using **StandardScaler**
- **PCA** is applied to reduce the feature space to **4 principal components**

### Model Architecture
The classification model is a simple feedforward neural network:

- Input layer: based on PCA-transformed feature size
- Hidden layer 1: **64 neurons + ReLU**
- Hidden layer 2: **64 neurons + ReLU**
- Output layer: **7 neurons** for multiclass prediction

### Loss and Optimizer
- Loss function: **CrossEntropyLoss**
- Optimizer: **Adam**

### Evaluation
The model is evaluated on the test set using:
- Accuracy from manual batch evaluation
- `accuracy_score` from scikit-learn

## 2) ANN Regression

The notebook `ann_reg.ipynb` implements a **regression model** for power plant data.

### Dataset
- Dataset used: **powerplant_data.csv**
- Target column: **`PE`**

Typical feature columns visible from the notebook:
- `AT`
- `V`
- `AP`
- `RH`

### Preprocessing Steps
- Null values are checked
- Features and target are separated
- Train-test split is performed
- Features are standardized using **StandardScaler**

### Model Architecture
The regression ANN is a small fully connected network:

- Input layer: number of input features
- Hidden layer 1: **6 neurons + ReLU**
- Hidden layer 2: **6 neurons + ReLU**
- Output layer: **1 neuron**

### Loss and Optimizer
- Loss function: **MSELoss**
- Optimizer: **Adam**

### Training Logic
- Model is trained for **100 epochs**
- Training loss and validation loss are tracked
- Best model weights are saved as `best_model.pt`

### Evaluation
The notebook evaluates regression performance using:
- Training MSE
- Testing MSE
- Loss curve visualization

## 3) Streamlit App

The file `app.py` provides a simple interactive UI for the **Power Plant ANN Regression** model.

### App Features
- Upload a CSV file
- Train the ANN model from the interface
- Adjust:
  - epochs
  - batch size
  - learning rate
- View:
  - dataset preview
  - missing values
  - test MSE
  - test R² score
  - training/validation loss curve
  - prediction table
- Download:
  - predictions CSV
  - trained model bundle
- Make manual predictions through form inputs

## Tech Stack

- **Python**
- **PyTorch**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Matplotlib**
- **Streamlit**

## Installation

Clone the repository:

```bash
git clone https://github.com/sriKritarth/ANN-classification.git
cd ANN-classification
```

Install dependencies:

```bash
pip install torch pandas numpy scikit-learn matplotlib streamlit
```

## How to Run

### Run the notebooks
Open Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Then open:
- `ann_classification.ipynb`
- `ann_reg.ipynb`

### Run the Streamlit app

```bash
streamlit run app.py
```

## Expected Input Files

Place the following datasets in the project directory before running the notebooks/app:

- `DateFruit_Dataset.csv`
- `powerplant_data.csv`

## Learning Highlights

This repository is useful for understanding:

- Difference between ANN classification and ANN regression
- Data preprocessing for neural networks
- Label encoding for categorical targets
- PCA-based dimensionality reduction
- Training PyTorch models using DataLoader
- Tracking train/validation loss
- Saving and reloading model weights
- Building a simple ML web app with Streamlit

## Future Improvements

Possible improvements for this project:

- Add a `requirements.txt` file
- Add dataset download/source links
- Add confusion matrix for classification
- Add MAE / RMSE metrics for regression
- Add early stopping
- Save classification model weights too
- Add better project modularization
- Deploy the Streamlit app online

## Note

This project appears to be built mainly for **practice and learning purposes**, and it is a good demonstration of implementing ANN workflows end-to-end for both major supervised learning tasks.

## Author

**Kritarth Srivastava**  
GitHub: [@sriKritarth](https://github.com/sriKritarth)
