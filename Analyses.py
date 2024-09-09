from setfit import SetFitModel
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import time
from requests.exceptions import ReadTimeout

# ========================== Original Model Performance ==========================

# Import Result Values
n_samples = [1, 3, 5, 10, 20, 30, 40, 50]
'''
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
'''
# ========================== Barplot Data ==========================
data = pd.read_csv('./data.csv', sep = ';')
pivot_df = data.pivot_table(index='repo', columns='label', aggfunc='size', fill_value=0)

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']

pivot_df.plot(kind='bar', stacked=True, color=['0.1', '0.4', '0.6'], figsize=(10, 6))

#plt.title(r'Distribution of Labels Across Repositories', fontsize=14, fontweight='bold')
plt.xlabel(r'Repositories', fontsize=12)
plt.ylabel(r'Count of Labels', fontsize=12)
plt.legend(title=r'Labels', fontsize=10, title_fontsize='11')
plt.grid(True, axis='y', which='both', linestyle='--', linewidth=0.6)
plt.tight_layout()

#plt.savefig('/Users/xeniamarlenezerweck/Documents/Verzeichnis/Master/Thesis/Graphics/Distribution_of_Labels.pdf', format='pdf')
plt.show()

# ========================== Percentage of Labels ==========================
label_counts = data['label'].value_counts()
total_labels = label_counts.sum()
label_percentages = (label_counts / total_labels) * 100
percentage_df = pd.DataFrame({'Count': label_counts, 'Percentage': label_percentages})
print(percentage_df)

# ========================== Classifier Performances ==========================
train_SVC = pd.read_csv('./train_dataset_SVC_long.csv')
X_train_SVC = train_SVC['sentences'].tolist()
y_train_SVC = train_SVC['label'].tolist()
train_GBC = pd.read_csv('./train_dataset_randomforest.csv')
X_train_GBC = train_GBC['sentences'].tolist()
y_train_GBC = train_GBC['label'].tolist()
train_LRC = pd.read_csv('./train_dataset_LogisticRegression.csv')
X_train_LRC = train_LRC['sentences'].tolist()
y_train_LRC = train_LRC['label'].tolist()

test_data_SVC = pd.read_csv('./test_dataset_SVC.csv')
texts_SVC = test_data_SVC['text'].tolist()
true_labels_SVC = test_data_SVC['label'].tolist()
test_data_GBC = pd.read_csv('./test_dataset_randomforest.csv')
texts_GBC = test_data_GBC['text'].tolist()
true_labels_GBC = test_data_GBC['label'].tolist()
test_data_LRC = pd.read_csv('./test_dataset_LogisticRegression.csv')
texts_LRC = test_data_LRC['text'].tolist()
true_labels_LRC = test_data_LRC['label'].tolist()

macro_f1s_SVC = [] # [0.20183308494783905, 0.19758257113467229, 0.24070206427296417, 0.22549879883501553, 0.2062320187668747, 0.2599262145803203, 0.5473956632975386, 0.5728886871756419]
macro_f1s_LRC = [] # [0.27009322247417483, 0.28993479114498505, 0.4287723375678884, 0.4442430954058861, 0.4437321937321938, 0.48802459695585393, 0.4876415041288709, 0.6230019877887802]
macro_f1s_GBC = [] # [0.3462924144969261, 0.3637006025395779, 0.42340948269413725, 0.4546413258723778, 0.4808960525687451, 0.43847801734103725, 0.5057574778279027, 0.5334360352975395]

# Import All Models
classifiers = ['SVC', 'LogisticRegression', 'GradientBoosting']

total_iterations = len(classifiers) * len(n_samples)
MAX_RETRIES = 3

with tqdm(total=total_iterations, desc="Processing Models") as pbar:
    for classifier in classifiers:
        for n_sample in n_samples:
            model_name = f'{classifier}_model_{n_sample}_samples_per_label'
            
            # Retry mechanism
            for attempt in range(MAX_RETRIES):
                try:
                    model = SetFitModel.from_pretrained(f'Catchy1282/{model_name}')
                    time.sleep(10)
                    break  # Break the loop if the model is successfully downloaded
                except ReadTimeout as e:
                    if attempt < MAX_RETRIES - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(5)  # Wait a bit before retrying
                    else:
                        print(f"Failed to download the model {model_name} after {MAX_RETRIES} attempts.")
                        continue  

            if classifier == 'SVC':
                model.fit(X_train_SVC, y_train_SVC, num_epochs=10)
                prediction = model.predict(texts_SVC)
                macro_f1s_SVC.append(f1_score(true_labels_SVC, prediction, average='macro'))

            if classifier == 'LogisticRegression':
                model.fit(X_train_LRC, y_train_LRC, num_epochs=10)
                prediction = model.predict(texts_LRC)
                macro_f1s_LRC.append(f1_score(true_labels_LRC, prediction, average='macro'))

            if classifier == 'GradientBoosting':
                model.fit(X_train_GBC, y_train_GBC, num_epochs=10)
                prediction = model.predict(texts_GBC)
                macro_f1s_GBC.append(f1_score(true_labels_GBC, prediction, average='macro'))

            pbar.update(1)
# ========================== Lineplot Classifier Performance ==========================
plt.figure(figsize=(10, 6))
plt.plot(n_samples, macro_f1s_SVC, label='Support Vector Classifier', marker='o', color='0.1')  # Dark grey
plt.plot(n_samples, macro_f1s_GBC, label='Gradient Boosted Classifier', marker='s', color='0.4')  # Medium grey
plt.plot(n_samples, macro_f1s_LRC, label='Logistic Regression Classifier', marker='^', color='0.6')  # Light grey

plt.xlabel(r'Number of Samples per Label', fontsize=12)
plt.ylabel(r'Performance (Macro F1 Score)', fontsize=12)
plt.xticks(n_samples)
plt.ylim(.1, .7)
plt.legend(title=r'Classifiers', fontsize=10, title_fontsize='11', loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.6)
plt.tight_layout()
plt.savefig('/Users/xeniamarlenezerweck/Documents/Verzeichnis/Master/Thesis/Graphics/Classifier_Performance.pdf', format='pdf')
plt.show()

# ========================== Import Best Models ==========================
# ========= SVC =========
SVC_model = SetFitModel.from_pretrained(f'Catchy1282/SVC_model_50_samples_per_label') # max = 50
SVC_model.fit(X_train_SVC, y_train_SVC, num_epochs = 10)

# ========= GBC =========
Gradient_model = SetFitModel.from_pretrained(f'Catchy1282/GradientBoosting_model_50_samples_per_label') # max = 50
Gradient_model.fit(X_train_GBC, y_train_GBC, num_epochs = 10)

# ========= LRC =========
LR_model = SetFitModel.from_pretrained(f'Catchy1282/LogisticRegression_model_50_samples_per_label') # max = 50
LR_model.fit(X_train_LRC, y_train_LRC, num_epochs = 10)

# ========================== In Depth Analysis of Classifiers ==========================
# ========= SVC =========
predicted_labels_SVC = SVC_model.predict(texts_SVC)

# Confusion Matrix
CM_SVC = confusion_matrix(true_labels_SVC, predicted_labels_SVC, normalize='all')
heatmap_SVC = sns.heatmap(CM_SVC, annot=True, fmt=".2f", cmap="Greys")

# F1
macro_f1_SVC = f1_score(true_labels_SVC, predicted_labels_SVC, average= 'macro') # 0.5728886871756419
f1_scores_SVC = f1_score(true_labels_SVC, predicted_labels_SVC, average=None)
print("Per-class F1 scores:", f1_scores_SVC) # Per-class F1 scores: [0.7372549  0.59272727 0.38868389]

# Accuracy, Precision and Recall
accuracy_score(true_labels_SVC, predicted_labels_SVC) # 0.6219822109275731
precision_score(true_labels_SVC, predicted_labels_SVC, average = None) # [0.92286115, 0.5, 0.2953271] = TP/(TP+FP)
recall_score(true_labels_SVC, predicted_labels_SVC, average = None) # [0.61380597, 0.72767857, 0.56834532] In an imbalanced dataset where the number of actual positives is very, very low, say 1-2 examples in total, recall is less meaningful and less useful as a metric

# ========= LRC =========
predicted_labels_LRC = LR_model.predict(texts_LRC)

# Confusion Matrix
CM_LRC = confusion_matrix(true_labels_LRC, predicted_labels_LRC, normalize='all')
heatmap_LRC = sns.heatmap(CM_LRC, annot=True, fmt=".2f", cmap="Greys")

# F1
macro_f1_LRC = f1_score(true_labels_LRC, predicted_labels_LRC, average= 'macro') # 0.6230019877887802
f1_scores_LRC = f1_score(true_labels_LRC, predicted_labels_LRC, average=None)
print("Per-class F1 scores:", f1_scores_LRC) # Per-class F1 scores: [0.83549161 0.62054507 0.41296928]

# Accuracy
accuracy_score(true_labels_LRC, predicted_labels_LRC) # 0.7242693773824651
precision_score(true_labels_LRC, predicted_labels_LRC, average = None) #[0.85982231, 0.58498024, 0.39285714]
recall_score(true_labels_LRC, predicted_labels_LRC, average = None) #[0.8125, 0.66071429, 0.4352518 ]

# ========= GBC =========
predicted_labels_GBC = Gradient_model.predict(texts_GBC)

# Confusion Matrix
CM_GBC = confusion_matrix(true_labels_GBC, predicted_labels_GBC, normalize='all')
heatmap_GBC = sns.heatmap(CM_GBC, annot=True, fmt=".2f", cmap="Greys")

# F1
macro_f1_GBC = f1_score(true_labels_GBC, predicted_labels_GBC, average= 'macro') # 0.583630804751229
f1_scores_GBC = f1_score(true_labels_GBC, predicted_labels_GBC, average=None)
print("Per-class F1 scores:", f1_scores_GBC) # Per-class F1 scores: [0.84985603 0.60759494 0.14285714]

# Accuracy
accuracy_score(true_labels_GBC, predicted_labels_GBC) # 0.6486658195679796
precision_score(true_labels_GBC, predicted_labels_GBC, average = None) # [0.8125    , 0.66071429, 0.4352518 ]
recall_score(true_labels_GBC, predicted_labels_GBC, average = None) # [0.9636194 , 0.53571429, 0.08273381]
