K-Nearest Neighbors (KNN) Model for Environmental Impact Assessment
This project implements a K-Nearest Neighbors (KNN) model to assess the environmental impact of war-related activities in Beirut, Lebanon. The model utilizes a simulated dataset containing two features: displaced population and infrastructure destroyed (in square kilometers), to classify the impact into three categories: Low, Moderate, and High.

Features
Displaced Population: Represents the number of individuals displaced due to conflict.
Infrastructure Destroyed (km²): Indicates the area of infrastructure destroyed during the conflict.
Usage
The KNN model is trained on a portion of the dataset and tested for accuracy, providing predictions for new data points to assess potential environmental impacts based on historical conflict data. The model's performance is evaluated using accuracy metrics, and predictions are made for hypothetical scenarios.

Requirements
Python
NumPy
scikit-learn