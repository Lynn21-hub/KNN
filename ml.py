import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#ceated hypothetical dataset 
data = np.array([
    [3000, 1, 0],   # Period 1: 1 km² destruction
    [12000, 5, 2],  # Period 2: 5 km² destruction
    [2000, 0, 0],   # Period 3: 0 km² destruction
    [20000, 10, 2], # Period 4: 10 km² destruction
    [4000, 2, 1],   # Period 5: 2 km² destruction
    [10000, 6, 2],  # Period 6: 6 km² destruction
    [1000, 0, 0],   # Period 7: 0 km² destruction
    [7000, 3, 1],   # Period 8: 3 km² destruction
    [15000, 8, 2],  # Period 9: 8 km² destruction
    [2500, 1, 1],   # Period 10: 1 km² destruction
    [9000, 4, 2],   # Period 11: 4 km² destruction
    [1500, 0, 0],   # Period 12: 0 km² destruction
    [18000, 9, 2],  # Period 13: 9 km² destruction
    [4000, 2, 1],   # Period 14: 2 km² destruction
    [8000, 3, 1],   # Period 15: 3 km² destruction
    [2000, 0, 0],   # Period 16: 0 km² destruction
    [11000, 7, 2],  # Period 17: 7 km² destruction
    [3500, 2, 1],   # Period 18: 2 km² destruction
    [1200, 0, 0],   # Period 19: 0 km² destruction
    [13000, 6, 2],  # Period 20: 6 km² destruction
])

X = data[:, :-1]  # Features: displaced_population and infrastructure_destroyed_km2
y = data[:, -1]   # Target: Environmental impact category

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

new_period = np.array([[8500, 3]])  # Hypothetical displaced population and infrastructure destruction
predicted_impact = knn.predict(new_period)
impact_labels = ["Low impact", "Moderate impact", "High impact"]
print(f"Predicted Environmental Impact: {impact_labels[predicted_impact[0]]}")
