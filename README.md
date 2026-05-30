# DevTown Machine Learning 
# Final Machine Learning Project: Customer Retail Analysis

## 📌 Project Objective
The objective of this project is to build, evaluate, and compare multiple supervised Machine Learning models using a customer retail dataset. This project demonstrates the complete end-to-end ML workflow, including data preprocessing, feature selection, categorical encoding, data visualization, model training, and performance evaluation.

---

## 📊 Dataset & Features
The project utilizes a customer retail dataset, focusing on the following key features for analysis and modeling:
* **Quantity**: The number of items purchased.
* **UnitPrice**: The price per item.
* **Country**: The geographic location of the customer (categorical).

---

## ⚙️ Project Workflow
1. **Data Loading**: Import and read the customer dataset using `pandas`.
2. **Data Cleaning**: Handle and drop missing values to ensure data quality.
3. **Categorical Encoding**: Transform the `Country` column into numerical values using `LabelEncoder`.
4. **Data Visualization**: Generate exploratory plots of the customer data using `matplotlib`.
5. **Data Splitting**: Partition the dataset into training and testing sets using `train_test_split`.
6. **Model Training**: Train three distinct classifiers:
   * Logistic Regression
   * Decision Tree Classifier
   * K-Nearest Neighbors (KNN)
7. **Model Evaluation**: Predict outcomes and evaluate each model using quantitative metrics.
8. **Performance Comparison**: Generate visual comparisons of the models' performances.

---

## 🧠 Machine Learning Models Used
The following classification algorithms are implemented and compared:
* **Logistic Regression** (`sklearn.linear_model`)
* **Decision Tree Classifier** (`sklearn.tree`)
* **K-Nearest Neighbors (KNN)** (`sklearn.neighbors`)

---

## 📈 Evaluation Metrics & Visualizations
### Evaluation Metrics
* **Accuracy Score**: To measure the percentage of correct predictions.
* **Confusion Matrix**: To analyze true positives, true negatives, false positives, and false negatives.

### Visualizations Included
* **Customer Distribution Graph**: Visualizing feature relationships (e.g., Quantity vs. UnitPrice).
* **Model Accuracy Comparison Graph**: A comparative chart illustrating the accuracy performance across all trained models.

---

## 🚀 Learning Outcomes
* Developed a deep understanding of the complete Machine Learning pipeline.
* Gained hands-on experience with foundational supervised learning algorithms.
* Mastered critical model evaluation techniques and interpretations.
* Learned to assess and contrast different ML algorithms through performance graphs and confusion matrices.

---

## 🎯 Conclusion
This project successfully demonstrates how different Machine Learning algorithms perform when trained on the same retail dataset. It highlights the critical role that data preprocessing, strategic feature selection, and comprehensive evaluation play when deploying machine learning to solve real-world analytical problems.
