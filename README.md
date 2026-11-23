# FUTURE_ML_02 – Support Ticket Classification and Prioritization

## Project Overview

This project was completed as part of the Future Interns Machine Learning Internship (Task 2).

The aim is to develop a machine learning system that automatically classifies customer support tickets into categories and predicts their priority levels.

The system uses Natural Language Processing (NLP) and machine learning techniques to help support teams organise incoming tickets and respond more efficiently.

## Objectives

- Classify support tickets into the correct support department.
- Predict ticket priority as High, Medium, or Low.
- Apply text preprocessing and TF-IDF feature extraction.
- Compare different machine learning algorithms.
- Evaluate model performance using classification metrics.

## Dataset

The project uses the Customer IT Support – Tickit dataset.

The selected dataset contains 20,000 support tickets in multiple languages. For this project, only English-language tickets were selected.

After cleaning and preprocessing, the working dataset contained 11,923 tickets.

The main features used were:
- Subject
- Body

The target variables were:
- Queue (ticket category)
- Priority (ticket priority)

## Data Preprocessing

The following steps were performed:

1. Loaded and explored the dataset.
2. Selected English-language tickets.
3. Handled missing subject and body values.
4. Combined subject and body into one text feature.
5. Removed empty ticket descriptions.
6. Converted text into numerical features using TF-IDF.

## Machine Learning Models

Three algorithms were compared:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes

Linear SVM achieved the strongest overall performance and was selected for the final models.

Class weighting was also tested to improve performance across different classes.

## Final Model Performance

### Ticket Category Classification

| Metric | Score |
|---|---:|
| Accuracy | 52.96% |
| Macro Precision | 64% |
| Macro Recall | 45% |
| Macro F1-score | 51% |

### Ticket Priority Prediction

| Metric | Score |
|---|---:|
| Accuracy | 58.95% |
| Macro Precision | 58% |
| Macro Recall | 58% |
| Macro F1-score | 58% |

## Results and Observations

The priority prediction model performed better than the category classification model.

The category model performed particularly well on Billing and Payments and Technical Support, but struggled with some less common categories.

The priority model showed more balanced performance across High, Medium, and Low priority classes.

The results demonstrate that machine learning can assist support teams, although further improvements are needed before using the models in a real production environment.

## Repository Structure

```text
FUTURE_ML_02/
│
├── README.md
├── support_ticket_classification.ipynb
└── models/
    ├── category_classifier.joblib
    └── priority_classifier.joblib
```

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Linear SVM
- Joblib
- Matplotlib
- Seaborn

## How to Use the Models

The trained models are saved in the `models` folder.

They can be loaded using Joblib:

```python
import joblib

category_model = joblib.load("models/category_classifier.joblib")
priority_model = joblib.load("models/priority_classifier.joblib")

new_ticket = [
    "I cannot access my account because of a login error."
]

category = category_model.predict(new_ticket)
priority = priority_model.predict(new_ticket)

print("Predicted Category:", category[0])
print("Predicted Priority:", priority[0])
```

Predictions should be treated as recommendations and reviewed by support staff.

## Conclusion

This project demonstrates how Natural Language Processing and machine learning can be applied to customer support ticket classification and prioritization.

The experiments showed that Linear SVM provided the best results among the tested algorithms. The project also highlighted the importance of data quality, class distribution, and model evaluation.

Further improvements could include testing additional algorithms, improving text preprocessing, and using more representative training data.

---

**Internship:** Future Interns  
**Task:** Machine Learning Task 2  
**Project:** Support Ticket Classification and Prioritization
