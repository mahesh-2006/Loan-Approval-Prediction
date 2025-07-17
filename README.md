# 🏦 Loan Approval Prediction

This project aims to build a predictive model to determine whether a loan application will be approved or not, based on historical applicant data. It uses various machine learning algorithms to achieve high accuracy and interpretability.

## 📌 Problem Statement

Financial institutions receive thousands of loan applications every day. Manually evaluating them can be time-consuming and error-prone. The goal of this project is to develop a machine learning model that can automatically predict whether a loan should be approved, helping streamline the loan approval process and minimize human error.

## ✅ Proposed Solution

We developed a classification model using historical loan application data. The dataset includes applicant information such as gender, income, credit history, education, and more. Various models were tested including Logistic Regression, Decision Tree, Random Forest, SVM, and KNN. The best performing model was selected based on evaluation metrics like accuracy, precision, recall, F1-score, and ROC AUC.

## ⚙️ System Approach

### Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

### System Requirements

- Python 3.7+
- Jupyter Notebook or any Python IDE
- Web browser to access the interface (optional for deployment)

## 🧠 Machine Learning Algorithm

- **Random Forest Classifier** was selected as the final model due to its balanced accuracy and generalization.
- The model achieved **~80.5% accuracy**.
- The trained model was saved using `.pkl` (Pickle) for future use.

## 🚀 Deployment

The model can be deployed using:
- A Flask/Django web interface
- Streamlit/Gradio for quick demos
- Cloud platforms for large-scale use

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Random Forest       | 80.5%    | 0.31      | 0.18   | 0.23     | 0.67    |
| Decision Tree       | 70.9%    | 0.17      | 0.21   | 0.19     | 0.50    |
| Logistic Regression | 63.8%    | 0.23      | 0.55   | 0.33     | 0.65    |

> Random Forest was selected due to highest accuracy and decent ROC AUC.

## 📝 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mahesh-2006/Loan-Approval-Prediction.git
   ```
2. Navigate into the project directory:
   ```bash
   cd Loan-Approval-Prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook or Python script to train or test the model.

## 🔮 Future Scope

- Add more features like customer salary history, number of dependents, etc.
- Deploy using a real-time web interface with user input.
- Integrate deep learning models for improved accuracy.
- Extend to multi-class predictions (e.g., low/medium/high loan risk).

## 📚 References

- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Loan Prediction Dataset – Kaggle](https://www.kaggle.com/datasets)
- Python for Data Science (AICTE LMS)
