from setfit import SetFitModel
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========================== General Model Performance ==========================

# Import Result Values
n_samples = [1, 3, 5, 10, 20, 30, 40, 50]
SVC_performances = [0.17795678011885466, 0.40030983096485634, 0.46292115964002584, 0.4127155655123904, 
                    0.5191894168469259, 0.5391068254846626, 0.5366664831299305, 0.5716833095873302]

GradientBoostingClassifier_performances = [0.3106745381703343, 0.41039541009045677, 0.4230810739777218, 0.4674327670126357, 
                                           0.48473460756875325, 0.49821409462245564,0.5272439995389141, 0.5909367814858419]

LogisticRegression_performances = [0.40029805116420974, 0.4049049337148445, 0.4480098573748999, 0.40798545224541427, 
                                   0.44940168071820824, 0.5388840494877835, 0.5622384186485875, 0.5807185593761462]

# Create dicts
SVC_dict = {n_samples[i]: SVC_performances[i] for i in range(len(n_samples))}
GradientBoostingClassifier_dict = {n_samples[i]: GradientBoostingClassifier_performances[i] for i in range(len(n_samples))}
LogisticRegression_dict = {n_samples[i]: LogisticRegression_performances[i] for i in range(len(n_samples))}

# ========================== Plot Lineplot ==========================
plt.figure(figsize=(10, 6))
plt.plot(n_samples, SVC_performances, label='SVC', marker='o', color='blue')
plt.plot(n_samples, GradientBoostingClassifier_performances, label='Gradient Boosting', marker='s', color='green')
plt.plot(n_samples, LogisticRegression_performances, label='Logistic Regression', marker='^', color='red')

# APA style adjustments
plt.title('Classifier Performances vs. Number of Samples', fontsize=14, fontweight='bold')
plt.xlabel('Number of Samples per Label', fontsize=12)
plt.ylabel('Performance (Macro F1 Score)', fontsize=12)
plt.xticks(n_samples)
plt.ylim(.1, .7)
plt.legend(title='Classifiers', fontsize=10, title_fontsize='11')
plt.grid(True, which='both', linestyle='--', linewidth=0.6)
plt.tight_layout()

plt.show()

# ========================== Import Best Models ==========================

SVC_model = SetFitModel.from_pretrained(f'Catchy1282/SVC_model_{max(SVC_dict, key=SVC_dict.get)}_samples_per_label') # max = 50
Gradient_model = SetFitModel.from_pretrained(f'Catchy1282/SVC_model_{max(GradientBoostingClassifier_dict, key=GradientBoostingClassifier_dict.get)}_samples_per_label') # max = 50
LR_model = SetFitModel.from_pretrained(f'Catchy1282/SVC_model_{max(LogisticRegression_dict, key=LogisticRegression_dict.get)}_samples_per_label') # max = 50







test_dataset = pd.read_csv('./test_dataset_LogisticRegression.csv')

texts = test_dataset['text'].tolist()  # Input texts for prediction
true_labels = test_dataset['label'].tolist()  # True labels
predicted_labels = model.predict(texts)

# Generate a confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()