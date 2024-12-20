# Heart-Disease-Prediction-and-Classification-Report-with-Graphs


This repository contains a machine learning project that uses the UCI Heart Disease dataset to predict the presence of heart disease in patients. The project utilizes various data cleaning techniques and applies a K-Nearest Neighbors (KNN) classifier for the prediction task.

## Dataset

The dataset used is the **UCI Heart Disease Dataset**, which includes patient data on factors such as age, sex, blood pressure, cholesterol, and other health indicators. The goal is to predict whether or not the patient has heart disease.

### Dataset Source:
- **UCI Heart Disease Dataset**: Available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

## Project Overview

The project involves the following steps:
1. **Data Preprocessing**:
   - Handling missing values (dropping rows or filling them with mean values).
   - Removing columns with too many missing values.
   - Converting data types as needed.
  
2. **Data Exploration**:
   - Basic statistical analysis of the dataset.
   - Visualization of distributions for relevant features.

3. **Model Training**:
   - Splitting the dataset into training and testing sets.
   - Using the K-Nearest Neighbors (KNN) algorithm to train a classifier for heart disease prediction.

4. **Evaluation**:
   - Evaluating the model using accuracy, confusion matrix, and classification report.

## Requirements

Make sure to install the required libraries before running the code:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. **Download the dataset** and place it in the project directory.
2. **Run the Python script** (e.g., `heart_disease_prediction.py`) to train the model and evaluate its performance.

```bash
python heart_disease_prediction.py
```

## File Structure

```
/Heart-Disease-Prediction
│
├── UCI_Heart_Disease_Dataset_Combined.csv  # Dataset file
├── heart_disease_prediction.py            # Python script for analysis and prediction
├── README.md                             # This README file
└── requirements.txt                      # Python dependencies
```

## Data Preprocessing Details

- **Handling Missing Values**: The dataset has missing values, which are handled in different ways:
  - Rows with missing values are dropped.
  - Columns with a significant number of missing values are removed.
  - Numerical columns with missing values are filled with their mean.

- **Data Splitting**: The dataset is split into training and test sets (80% training, 20% test) to evaluate the model performance.

## Model

### K-Nearest Neighbors Classifier

- The KNN algorithm is used to classify whether or not a patient has heart disease based on their features.
- The model is evaluated based on:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-Score)

## Results

The trained KNN model will provide an accuracy score, and the confusion matrix and classification report will give insights into the model's performance on the test set.

## Contributing

Feel free to fork this repository, submit issues, or create pull requests. Any contributions to improve the code or data processing steps are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Notes:
- Replace **`heart_disease_prediction.py`** with the name of your script file.
- Update or add any additional setup instructions if you have specific steps to follow.
