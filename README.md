# Resume Classification System

## Project Overview
This project classifies resumes into different job categories using Machine Learning and Natural Language Processing (NLP).

The system analyzes resume text and predicts the appropriate category such as:
- Data Science
- HR
- Web Development
- Java Developer
- Testing
- DevOps

---

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF Vectorizer
- Naive Bayes Classifier

---

## Dataset
The project uses a resume dataset containing:
- Resume text
- Job category labels

Dataset Source:
Kaggle Resume Dataset

---

## Project Workflow

1. Load Dataset
2. Text Preprocessing
3. TF-IDF Vectorization
4. Train-Test Split
5. Model Training using Naive Bayes
6. Resume Category Prediction
7. Accuracy Evaluation

---

## Machine Learning Model
Algorithm Used:
- Multinomial Naive Bayes

Reason:
- Efficient for text classification problems
- Fast and accurate for NLP tasks

---

## Accuracy
The model achieved approximately:

```text
98% Accuracy
```

---

## Sample Prediction

Input Resume Skills:

```text
Python, SQL, Machine Learning, Deep Learning
```

Predicted Category:

```text
Data Science
```

---

## How to Run the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Program

```bash
python app.py
```

---

## Project Structure

```text
Resume-Classification-System
│
├── app.py
├── README.md
├── requirements.txt
└── dataset
```

---

## Future Improvements
- Web interface using Flask
- PDF resume upload
- Deep Learning models
- Better UI

---

## Author
Bhawansh Garg