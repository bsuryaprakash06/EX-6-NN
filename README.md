| **Name** | **Register Number** | **Experiment No.** | **Date** |
|-----------|---------------------|--------------------|----------|
| Surya Prakash B | 212224230281 | 6 | 28 - 08 - 2026 |

# Heart Attack Prediction Using MLP

### Aim:

To construct a Multi-Layer Perceptron to predict heart attack using Python.

### Algorithm:

**Step 1:** Import the required libraries: `numpy`, `pandas`, `MLPClassifier`, `train_test_split`, `StandardScaler`, `accuracy_score`, `f1_score`, `recall_score`, `classification_report`, `confusion_matrix`, and `matplotlib.pyplot`.

**Step 2:** Load the heart disease dataset from a file using `pd.read_csv()`.

**Step 3:** Separate the features and labels from the dataset using `data.iloc` values for features (`X`) and `data.iloc[:, -1].values` for labels (`y`).

**Step 4:** Split the dataset into training and testing sets using `train_test_split()`.

**Step 5:** Normalize the feature data using `StandardScaler()` to scale the features to have zero mean and unit variance.

**Step 6:** Create an `MLPClassifier` model with the desired architecture and hyperparameters, such as `hidden_layer_sizes`, `max_iter`, and `random_state`.

**Step 7:** Train the MLP model on the training data using `mlp.fit(X_train, y_train)`. The model adjusts its weights and biases iteratively to minimize the training loss.

**Step 8:** Make predictions on the testing set using `mlp.predict(X_test)`.

**Step 9:** Calculate the accuracy by comparing the predicted labels (`y_pred`) with the actual labels (`y_test`) using `accuracy_score()`.

**Step 10:** Calculate the F1-Score using `f1_score()`.

**Step 11:** Calculate the Recall-Score using `recall_score()`.

**Step 12:** Generate the Classification Report using `classification_report()`, which displays precision, recall, F1-score, and support for each class.

**Step 13:** Generate the Confusion Matrix using `confusion_matrix()` to determine the number of correct and incorrect predictions for each class.

**Step 14:** Display the Accuracy, F1-Score, Recall-Score, and Classification Report.

**Step 15:** Plot the Confusion Matrix using `matplotlib.pyplot`.

**Step 16:** Plot the error convergence during training using `mlp.loss_curve_`, `plt.plot()`, and `plt.show()`.

### Program:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load the dataset
data = pd.read_csv("heart.csv")

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display results
print("MLP Heart Disease Prediction")
print("--------------------------------")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("Hidden Layers    : (16, 8)")
print("Activation       : ReLU")

print("\nEvaluation Metrics")
print("--------------------------------")
print("Accuracy  :", round(accuracy * 100, 2), "%")
print("F1-Score  :", round(f1, 4))
print("Recall    :", round(recall, 4))

print("\nClassification Report")
print("--------------------------------")
print(report)

print("Confusion Matrix")
print("--------------------------------")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()
plt.title("Confusion Matrix - MLP")
plt.show()

# Plot training loss convergence
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss Convergence")
plt.grid(True)
plt.show()
```

## Output:

### Metrics:
<img width="289" height="394" alt="{380057E5-5A4D-43AD-BFA3-5850523E210F}" src="https://github.com/user-attachments/assets/3ee0cbfc-4250-4930-9006-20a74b8a5269" />

### Confusion - Matrix:
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/421741cb-5df0-4a5f-80bb-3bf3d1402c46" />

### MLP Training Loss Convergence:
<img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/781722f8-4323-4a4f-a783-b67358f55a3f" />

### Result:
Thus, an Artificial Neural Network using a Multi-Layer Perceptron (MLP) was successfully constructed and trained to predict heart disease using Python. The model was successfully evaluated using **Accuracy, F1-Score, Recall-Score, Classification Report, and Confusion Matrix**. The training loss convergence was also plotted to analyze the learning performance of the MLP model.
