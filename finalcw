#use version pythin vesrsion 3.9.1.1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import chi2_contingency


import shap
import os

print(tf.__version__)

#hide gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#restore
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


#load csv
#rawData = pd.read_csv("diabetic_data.csv")
#Changed file directory - Best
rawData = pd.read_csv("CW\diabetic_data.csv")

#show shape
print("CSV READ: ")
print(rawData.shape)

#replace ? with numpy.nan
rawData.replace('?', np.nan, inplace=True)
#threshold for dropping cols
threshold = len(rawData) / 2

##some feature engineering: Count the number of diabetes medications a patient is on. This could indicate the severity of their diabetes.
medications = ['metformin', 'repaglinide','nateglinide', 'chlorpropamide', 'glimepiride','acetohexamide', 'glipizide', 'glyburide','tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone','metformin-rosiglitazone', 'metformin-pioglitazone']
rawData['diabetes_medication_count'] = rawData[medications].apply(lambda x: x != 'No').sum(axis=1)
#Count the number of increases/decreases and steadies
rawData['medications_increased'] = rawData[medications].apply(lambda x: (x == 'Up').sum(), axis=1)
rawData['medications_decreased'] = rawData[medications].apply(lambda x: (x == 'Down').sum(), axis=1)
#rawData['medications_held'] = rawData[medications].apply(lambda x: (x == 'Steady').sum(), axis=1)
rawData['total_medication_changes'] = rawData['medications_increased'] + rawData['medications_decreased']

np.random.seed(42)
missing_race = rawData['race'].isnull()
rawData.loc[missing_race, 'race'] = np.random.choice(
    rawData['race'].dropna().unique(),
    size=len(rawData[missing_race])
)


#drop cols where >50% of data is ?
rawData.dropna(axis=1, thresh=threshold, inplace=True)
#drop cols where data is >95% the same
for column in rawData.columns:
    most_common_pct = (rawData[column].value_counts(normalize=True).iloc[0]) * 100
    if most_common_pct > 95:
        rawData.drop(column, axis=1, inplace=True)
print("Drop 95% same cols and cols where >50% of data is gone: ")
print(rawData.shape)


#age transformation method
def get_middle_age(age_range):
    start, end = age_range[1:-1].split("-")  # Remove brackets and split by '-'
    middle_age = (int(start) + int(end)) / 2
    return middle_age

#applying to data
rawData['age'] = rawData['age'].apply(get_middle_age)
#print(rawData.head())

# def map_age_to_group(age):  #HAS A BIG EFFECT ON THE MODEL - REDUCES PREC INCREASES REC
#     if age < 18:
#         return 'Pediatric'
#     elif age < 65:
#         return 'Adult'
#     else:
#         return 'Senior'
# rawData['age_group'] = rawData['age'].apply(map_age_to_group)


#avoid data leakage by just keeping one
rawData = rawData.drop_duplicates(subset='patient_nbr', keep='first')

#hm
#rawData['medical_burden'] = rawData['num_lab_procedures'] + rawData['num_procedures'] + rawData['num_medications'] + rawData['number_inpatient']+ rawData['number_emergency']+ rawData['number_outpatient']
#performs around the same
#rawData['medical_burden'] = rawData['num_lab_procedures'] + rawData['num_procedures'] + rawData['num_medications'] + rawData['number_inpatient']+ rawData['number_emergency']
rawData['medical_burden'] = np.log1p(rawData['num_lab_procedures']) + \
                            np.log1p(rawData['num_procedures']) + \
                            np.log1p(rawData['num_medications']) + \
                            np.log1p(rawData['number_inpatient']) + \
                            np.log1p(rawData['number_emergency'])

#diag selection and replacing all missing values with 0
columns_to_replace = ['diag_1', 'diag_2', 'diag_3']
rawData[columns_to_replace] = rawData[columns_to_replace].fillna(0)

#mgs and ac1 -ACTUALLY, the abscense of such tests might be risky towards readmission
#rawData['max_glu_serum'] = rawData['max_glu_serum'].replace('None', np.nan, inplace=True) 
#rawData['AC1Result'] = rawData['AC1Result'].replace('None', np.nan, inplace=True)

#### WE ARE CURRENTLY DROPPING ALOT OF DATA AS BOTH 'PAYER_CODE' AND 'MEDICAL_SPECIALITY' HAS ALOT OF EMPTY DATA ####
#### TO BE PRECISE: payer_code: 39.56% AND medical_specialty: 49.08% is empty ####
#### DATA IS DROPPED FROM (101766, 33) to (27140, 33) ####

rawData.drop('medical_specialty', axis=1, inplace=True)
rawData.drop('payer_code', axis=1, inplace=True)
#rawData.drop('')

#then drop any rows with missing values
rawData.dropna(inplace=True)
#print(rawData.shape)
#print(rawData.dtypes)


#numerical columns
numerical_columns = [
    'age', #exclude_columns below
    'time_in_hospital', 'num_lab_procedures', 
    'num_procedures', 'num_medications', 'number_outpatient', 
    'number_emergency', 'number_inpatient', 'number_diagnoses'
]

#categorical columns
categorical_columns = [
    'race', 'gender', 'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult', 'metformin',  'patient_nbr',
    'glimepiride',  'glipizide', 'glyburide', 
    'pioglitazone', 'rosiglitazone',  
    'insulin', 'change', 'diabetesMed', 'readmitted'
]

exclude_columns = ['encounter_id', 'patient_nbr'] #as its about the visit+the patient not just the patient?

#the data we want
all_columns = numerical_columns + categorical_columns
#categoricalData = rawData[categorical_columns]
#numericalData = rawData[numerical_columns]
rawData[numerical_columns] = rawData[numerical_columns].apply(pd.to_numeric, errors='coerce')

#numericalData.to_csv('numerical_data.csv', index=False)

#removing outliers 3std away from the mean 
means = rawData[numerical_columns].mean()
stds = rawData[numerical_columns].std()
mask = (np.abs(rawData[numerical_columns] - means) <= 3 * stds).all(axis=1)
filtered_data = rawData[mask]
print(filtered_data.shape)
#filtered_data = filtered_data.drop_duplicates(subset=['patient_nbr'], keep='first')
print(filtered_data.shape)
#print(filteredCategorical_data.shape)
#filtered_data.to_csv('filtered_data.csv', index=False)

##########################################################################################

filtered_data.loc[:, 'readmitted_binary'] = (filtered_data['readmitted'] != 'NO').astype(int) #important line for rest of code WE ARE TRYING TO PREDICT IF PEOPLE WILL BE READMITTED WITHIN LESS THAN 30 DAYS

########################################PART 2############################################


#BAR CHART FOR FREQUENCY OF READMISSION BY AGE

age_group_totals = filtered_data.groupby('age').size()
age_readmission_totals = filtered_data.groupby('age')['readmitted_binary'].sum()
age_readmission_rate = age_readmission_totals / age_group_totals

# Plot for Age
plt.figure(figsize=(10, 6))
plt.bar(age_readmission_rate.index, age_readmission_rate.values, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Rate of Readmission')
plt.title('Rate of Readmission by Age')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate the readmission rate by race
race_group_totals = filtered_data.groupby('race').size()
race_readmission_totals = filtered_data.groupby('race')['readmitted_binary'].sum()
race_readmission_rate = race_readmission_totals / race_group_totals

# Plot for Race
plt.figure(figsize=(10, 6))
plt.bar(race_readmission_rate.index, race_readmission_rate.values, color='skyblue')
plt.xlabel('Race')
plt.ylabel('Rate of Readmission')
plt.title('Rate of Readmission by Race')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate the readmission rate by gender
filtered_gender = filtered_data[filtered_data['gender'].isin(['Male', 'Female'])]
gender_group_totals = filtered_gender.groupby('gender').size()
gender_readmission_totals = filtered_gender.groupby('gender')['readmitted_binary'].sum()
gender_readmission_rate = gender_readmission_totals / gender_group_totals

# Plot for Gender
plt.figure(figsize=(10, 6))
plt.bar(gender_readmission_rate.index, gender_readmission_rate.values, color='skyblue')
plt.xlabel('Gender')
plt.ylabel('Rate of Readmission')
plt.title('Rate of Readmission by Gender')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#BAR CHART FOR DIAGS
diag_cols = ['diag_1', 'diag_2', 'diag_3']
melted_data = pd.melt(filtered_data, id_vars=['readmitted_binary'], value_vars=diag_cols, var_name='Diagnosis', value_name='Code')
diag_readmission_frequency = melted_data.groupby('Code')['readmitted_binary'].sum().reset_index().sort_values(by='readmitted_binary', ascending=False)
#WE NEED 3 BAR CHARTS - ONE FOR DIAG1 CONVERTED TO DISEASE NAME - THEN FOR 2 AND 3

#https://raw.githubusercontent.com/drobbins/ICD9/master/icd9.txt
categories = {
    'Infectious/Parasitic Diseases': range(1, 140),
    'Neoplasms': range(140, 240),
    'Endocrine/Metabolic Diseases': range(240, 250),
    # #DIABETES TYPES
    # 'Diabetes Mellitus': range(250, 250.1),
    # 'Diabetes With ketoacidosis': range(250.1, 250.2),
    # 'Diabetes With Hyperosmolarity': range(250.2, 250.3),
    # 'Diabetes With Coma': range(250.3, 250.4),
    # 'Diabetes With Renal Manifestations': range(250.4, 250.5),
    # 'Diabetes With Opthalmic Manifestations': range(250.5, 250.6),
    # 'Diabetes With Neurological Manifestations': range(250.6, 250.7),
    # 'Diabetes With Peripheral Circulatory Manifestations': range(250.7, 250.8),
    # 'Diabetes With Other Specified Manifestations': range(250.8, 250.9),
    # 'Diabetes With Unspecified Complication': range(250.9, 251),
    # #END OF DIABETES TYPES
    'Endocrine/Metabolic Diseases': range(251, 280),
    'Blood/Blood-Forming Organ Diseases': range(280, 290),
    'Mental and Behavioural Disorders': range(290, 320),
    'Nervous System Diseases': range(320, 360),
    'Eye Diseases': range(360, 380),
    'Ear/Mastoid Diseases': range(380, 390),
    'Circulatory System Diseases': range(390, 460),
    'Respiratory System Diseases': range(460, 520),
    'Digestive System Diseases': range(520, 580),
    'Skin/Subcutaneous Diseases': range(680, 710),
    'Musculoskeletal/Connective Tissue Diseases': range(710, 740),
    'Genitourinary System Diseases': range(580, 630),
    'Pregnancy/Childbirth': range(630, 680),
    'Perinatal Period Conditions': range(760, 780),
    'Congenital Abnormalities': range(740, 760),
    'Symptoms/Signs/Abnormal Findings': range(780, 800),
    'Injury/Poisoning': range(800, 1000),
    'External Causes of Morbidity and Mortality': range(800, 1000),
    'Health Status Factors and Health Services Contact': range(1000, 1100),
    # Adding a general "Other" category for diseases not fitting into the above categories
    'Other': range(1100, 10000)
}
v_code_categories = {
    'Vaccinations and Prophylactic Measures': range(1, 8),  # V01-V07
    'Infectious Diseases Status': range(8, 10),  # V08-V09
    'Personal and Family Health History': range(10, 20),  # V10-V19
    'Pregnancy and Reproductive Services': range(20, 30),  # V20-V29
    'Liveborn Infants and Birth Outcomes': range(30, 40),  # V30-V39
    'Mental and Behavioral Problems': [40],  # V40
    'Sensory/Transplant Status': range(41, 50),  # V41-V49
    'Surgical Aftercare and Other Postprocedural States': range(50, 60),  # V50-V59
    'Health Service Encounters': range(60, 70),  # V60-V69
    'Medical Examinations': range(70, 80),  # V70-V79
    'Special Screenings for Diseases and Disorders': range(80, 92)  # V80-V91
}

def map_diag_to_category(diag_code):
    try:
        diag_code_float = float(diag_code) #in case of decimals
        if(250<=diag_code_float<251): #had to do this here as range() cant handle floats
            if(250<=diag_code_float<250.1):
                return 'Diabetes Mellitus Without Mention of Complication'
            elif (250.1<=diag_code_float<250.2):
                return 'Diabetes With ketoacidosis'
            elif 250.2 <= diag_code_float < 250.3:
                return 'Diabetes with hyperosmolarity'
            elif 250.3 <= diag_code_float < 250.4:
                return 'Diabetes with other coma'
            elif 250.4 <= diag_code_float < 250.5:
                return 'Diabetes with renal manifestations'
            elif 250.5 <= diag_code_float < 250.6:
                return 'Diabetes with ophthalmic manifestations'
            elif 250.6 <= diag_code_float < 250.7:
                return 'Diabetes with neurological manifestations'
            elif 250.7 <= diag_code_float < 250.8:
                return 'Diabetes with peripheral circulatory disorders'
            elif 250.8 <= diag_code_float < 250.9:
                return 'Diabetes with other specified manifestations'
            elif 250.9 <= diag_code_float < 251:
                return 'Diabetes with unspecified complication'
        diag_code_int = int(diag_code_float)
        for category, icd_range in categories.items():
            if diag_code_int in icd_range:
                return category
    except ValueError:
        if not diag_code.startswith('V'):
            return 'Other'
        try:
            code_num = int(diag_code[1:])
            for category, code_range in v_code_categories.items():
                if code_num in code_range:
                    return category
        except ValueError:
            return 'Invalid V-code format'
    
    return 'Other Health Services Encounters'


for column in filtered_data[diag_cols]:
    filtered_data[column + '_category'] = filtered_data[column].apply(map_diag_to_category)

diag_1_freq = filtered_data['diag_1_category'].value_counts().reset_index().rename(columns={'index': 'Category', 'diag_1_category': 'Frequency'})
diag_1_freq.columns = ['Category', 'Frequency'] #necessary or keyerror 'Category'
diag_2_freq = filtered_data['diag_2_category'].value_counts().reset_index().rename(columns={'index': 'Category', 'diag_2_category': 'Frequency'})
diag_2_freq.columns = ['Category', 'Frequency']
diag_3_freq = filtered_data['diag_3_category'].value_counts().reset_index().rename(columns={'index': 'Category', 'diag_3_category': 'Frequency'})
diag_3_freq.columns = ['Category', 'Frequency']

def plot_frequency(df, title):
    #print(df.columns)
    plt.figure(figsize=(10, 8))
    plt.barh(df['Category'], df['Frequency'], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Category')
    plt.title(title)
    plt.tight_layout()
    #plt.show()

plot_frequency(diag_1_freq, 'Frequency of Categories in Primary Diagnosis (diag_1)')
plot_frequency(diag_2_freq, 'Frequency of Categories in Secondary Diagnosis (diag_2)')
plot_frequency(diag_3_freq, 'Frequency of Categories in Tertiary Diagnosis (diag_3)')


diag_1_readmission_rates = filtered_data.groupby('diag_1_category')['readmitted_binary'].mean().reset_index().rename(columns={'readmitted_binary': 'Readmission Rate'})
diag_2_readmission_rates = filtered_data.groupby('diag_2_category')['readmitted_binary'].mean().reset_index().rename(columns={'readmitted_binary': 'Readmission Rate'})
diag_3_readmission_rates = filtered_data.groupby('diag_3_category')['readmitted_binary'].mean().reset_index().rename(columns={'readmitted_binary': 'Readmission Rate'})

def plot_readmission_rates(df, title, threshold):
    # Filter the DataFrame to display only rates higher than the threshold
    df_filtered = df[df['Readmission Rate'] > threshold]
    # Sort the DataFrame by 'Readmission Rate' in descending order
    sorted_df = df_filtered.sort_values(by='Readmission Rate', ascending=False)
    plt.figure(figsize=(10, 8))
    category_column = sorted_df.columns[0]  # Dynamically get the category column name
    plt.barh(sorted_df[category_column], sorted_df['Readmission Rate'], color='skyblue')
    plt.xlabel('Readmission Rate')
    plt.ylabel('Diagnosis Category')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plotting the readmission rates
plot_readmission_rates(diag_1_readmission_rates, 'Readmission Rate by Primary Diagnosis Category', 0.35)
plot_readmission_rates(diag_2_readmission_rates, 'Readmission Rate by Secondary Diagnosis Category', 0.35)
plot_readmission_rates(diag_3_readmission_rates, 'Readmission Rate by Tertiary Diagnosis Category', 0.35)

age_table = pd.crosstab(filtered_data['age'], filtered_data['readmitted_binary'])
chi2, p_age, dof, expected = chi2_contingency(age_table)
print(f"Chi-square test for age: p={p_age}")
n = age_table.sum().sum()
k = min(age_table.shape)   #i think k can just equal 2 here? since its a binary outcome of 1 or 0
V = np.sqrt(chi2 / (n * (k - 1)))
print(f"Cramer's V for effect size between age: {V}")

gender_table = pd.crosstab(filtered_data['gender'], filtered_data['readmitted_binary'])
chi2, p_gender, dof, expected = chi2_contingency(gender_table)
print(f"Chi-square test for gender: p={p_gender}")
n = gender_table.sum().sum()
k = min(gender_table.shape)
V = np.sqrt(chi2 / (n * (k - 1)))
print(f"Cramer's V for effect size between genders: {V}")

race_table = pd.crosstab(filtered_data['race'], filtered_data['readmitted_binary'])
chi2, p_race, dof, expected = chi2_contingency(race_table)
print(f"Chi-square test for race: p={p_race}")
n = race_table.sum().sum()
k = min(race_table.shape)
V_race = np.sqrt(chi2 / (n * (k - 1)))
print(f"Cramer's V for effect size between races: {V_race}")

########################################PART 3############################################
model_columns = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 'number_inpatient', 'age', 'num_lab_procedures', 'number_diagnoses', 'num_procedures', 'readmitted_binary']
model_df = filtered_data[model_columns]

X = model_df.drop('readmitted_binary', axis=1)
y = model_df['readmitted_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

y_train_pred_rf = rf_model.predict(X_train)
confusion_matrix_train_rf = confusion_matrix(y_train, y_train_pred_rf)
accuracy_train_rf = accuracy_score(y_train, y_train_pred_rf)
precision_train_rf = precision_score(y_train, y_train_pred_rf)
recall_train_rf = recall_score(y_train, y_train_pred_rf)
f1_train_rf = f1_score(y_train, y_train_pred_rf)
roc_auc_train_rf = roc_auc_score(y_train, y_train_pred_rf)

confusion_matrix_test_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_test_rf = accuracy_score(y_test, y_pred_rf)
precision_test_rf = precision_score(y_test, y_pred_rf)
recall_test_rf = recall_score(y_test, y_pred_rf)
f1_test_rf = f1_score(y_test, y_pred_rf)
roc_auc_test_rf = roc_auc_score(y_test, y_pred_rf)

print("Random Forest Training Data Stats:")
print("Confusion Matrix:")
print(confusion_matrix_train_rf)
print(f"Accuracy: {accuracy_train_rf}")
print(f"Precision: {precision_train_rf}")
print(f"Recall: {recall_train_rf}")
print(f"F1 Score: {f1_train_rf}")
print(f"ROC AUC Score: {roc_auc_train_rf}\n")

print("Random Forest Validation Data Stats:")
print("Confusion Matrix:")
print(confusion_matrix_test_rf)
print(f"Accuracy: {accuracy_test_rf}")
print(f"Precision: {precision_test_rf}")
print(f"Recall: {recall_test_rf}")
print(f"F1 Score: {f1_test_rf}")
print(f"ROC AUC Score: {roc_auc_test_rf}")

#CrossValidation Scores
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nCross-validation scores:", cv_scores)
cv_accuracy = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
mean_cv_accuracy = cv_accuracy.mean()
print(f"Mean CV Accuracy: {mean_cv_accuracy}")
###########################################################################################

print(filtered_data.dtypes)

filtered_data.to_csv('filtered_data.csv', index=False)

def normalise(df):
    num_cols=df.select_dtypes(include=[np.number]).copy()
    df_norm=((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    return df_norm 


#these things have nothing to do with our data anymore. diags have been categorised and the rest are just useless
filtered_data =filtered_data.drop('readmitted', axis=1, )
filtered_data =filtered_data.drop('encounter_id', axis=1, )
filtered_data =filtered_data.drop('patient_nbr', axis=1, )
filtered_data =filtered_data.drop('diag_1', axis=1, ) #since weve categorised them
filtered_data =filtered_data.drop('diag_2', axis=1, )
filtered_data =filtered_data.drop('diag_3', axis=1, )
filtered_data =filtered_data.drop('age', axis=1, ) #for some reason model is more accurate without


#potential data leakage? as this has a direct connection to patient status post-visit, which could imply information about readmission.
#filtered_data =filtered_data.drop('discharge_disposition_id', axis=1)

#disproportionately significant?
#filtered_data =filtered_data.drop('number_inpatient', axis=1) 


print(filtered_data.shape)
filtered_data_encoded = pd.get_dummies(filtered_data, columns=['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'race', 'gender', 'metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin', 'change', 'diabetesMed', 'diag_1_category', 'diag_2_category', 'diag_3_category'])

df_norm = normalise(filtered_data_encoded)
filtered_data[df_norm.columns] = df_norm


#print(filtered_data_encoded.dtypes)

X = filtered_data_encoded.drop('readmitted_binary', axis=1)  # or any other target variable column
print(X.columns)
y = filtered_data_encoded['readmitted_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42) #oversampling
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# rus = RandomUnderSampler(random_state=42) #undersampling
# X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

#print(X_train_rus.shape) 

print(X_train_smote.shape)

########################################

# wcss = [] #elbow
# k_range = range(2, 21)  
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
#     kmeans.fit(X_train_smote)  
#     wcss.append(kmeans.inertia_)

# plt.figure(figsize=(10, 6))
# plt.plot(k_range, wcss, marker='o')
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('WCSS')
# plt.xticks(k_range)
# plt.show()

kmeans = KMeans(n_clusters=6)  #6 based on above elbow method
cluster_labels = kmeans.fit_predict(X_train_smote)
print('J-score = ', kmeans.inertia_)
print(kmeans.labels_)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X_train_smote)

#plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', s=1)
plt.xlabel('pca 1')
plt.ylabel('pca 2')
plt.title('Cluster visualization with PCA')
plt.colorbar()
plt.show()
full_dataset_clusters = kmeans.labels_

#############################################


classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_smote)
class_weightsB4 = compute_class_weight('balanced', classes=classes, y=y_train)
print("Class weights before smote: ", class_weightsB4)
print("Class weights after smote: ", class_weights)


print('\n ############################################### \n TENSORFLOW \n')

#custom F1 score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

#######

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
                  'accuracy',
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall'),
                  tf.keras.metrics.AUC(name='auc'),
                  F1Score()
              ])



num_folds = 5
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
validation_losses = []
validation_accuracies = []
validation_data_per_fold = []

features_all_folds = []  


if not os.path.exists('models'):
    os.makedirs('models')


for fold, (train_indices, val_indices) in enumerate(kfold.split(X_train_scaled, y_train_smote)):
    print(f'Fold {fold+1}/{num_folds}:')
    model = tf.keras.models.Sequential([ #re initialise every time
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    F1Score()
                ])

    checkpoint_filepath = f'models/best_model_fold_{fold+1}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_f1_score',
        mode='max',
        save_best_only=True)

    X_train_fold, X_val_fold = X_train_scaled[train_indices], X_train_scaled[val_indices]
    y_train_fold, y_val_fold = y_train_smote[train_indices], y_train_smote[val_indices]

    #train the model on the training data for this fold
    history = model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=64, validation_data=(X_val_fold, y_val_fold), verbose=0, callbacks=[model_checkpoint_callback])
    validation_data_per_fold.append((X_val_fold, y_val_fold))

    #evaluate the model
    eval_results = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    loss = eval_results[0]  
    accuracy = eval_results[1]  
    precision = eval_results[2]
    recall = eval_results[3]
    auc= eval_results[4]
    f1 = eval_results[5]

    train_eval_results = model.evaluate(X_train_fold, y_train_fold, verbose=0)

    print(f'Training loss for this fold: {train_eval_results[0]}, accuracy: {train_eval_results[1]}, precision: {train_eval_results[2]}, recall: {train_eval_results[3]}, AUC: {train_eval_results[4]}, F1 score: {train_eval_results[5]}')

    print(f'Validation loss for this fold: {loss}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, AUC: {auc}, F1 score: {f1} \n')


    validation_losses.append(loss)
    validation_accuracies.append(accuracy)

mean_validation_loss = np.mean(validation_losses)
mean_validation_accuracy = np.mean(validation_accuracies)
std_validation_loss = np.std(validation_losses)
std_validation_accuracy = np.std(validation_accuracies)

print(f'\nMean validation loss: {mean_validation_loss} (±{std_validation_loss})')
print(f'\nMean validation accuracy: {mean_validation_accuracy} (±{std_validation_accuracy})')


best_fold = np.argmax(validation_accuracies) + 1
best_checkpoint_filepath = f'models/best_model_fold_{best_fold}.h5'

#best_checkpoint_filepath = f'models/goodModels/70a70p70r80auc70f.h5'

model.load_weights(best_checkpoint_filepath)

X_train_smote = X_train_smote.astype('float32')
predictions = model.predict(X_train_smote).flatten()

eval_results = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Loss: {eval_results[0]}, Accuracy: {eval_results[1]}, Precision: {eval_results[2]}, Recall: {eval_results[3]}, AUC: {eval_results[4]}, F1 Score: {eval_results[-1]}")


data_with_clusters_and_predictions = pd.DataFrame(X_train_smote)  
data_with_clusters_and_predictions['Cluster'] = cluster_labels
data_with_clusters_and_predictions['AtRiskPrediction'] = predictions

cluster_analysis = data_with_clusters_and_predictions.groupby('Cluster')['AtRiskPrediction'].agg(['mean', 'count']).reset_index()
print(cluster_analysis)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5, s=1)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Cluster visualization with PCA and Flagged Patients')

#plot flagged patients
flagged_indices = np.where(predictions >= 0.8)[0]  
plt.scatter(reduced_features[flagged_indices, 0], reduced_features[flagged_indices, 1], color='red', label='Flagged Patients', s=1)

plt.colorbar()
plt.legend()
plt.show()

data_with_predictions = data_with_clusters_and_predictions
data_with_predictions=data_with_predictions.drop('Cluster', axis=1)

high_risk_patients = data_with_predictions[data_with_predictions['AtRiskPrediction'] > 0.8]

low_risk_patients = data_with_predictions[data_with_predictions['AtRiskPrediction'] < 0.4]

high_risk_patients = high_risk_patients.drop('AtRiskPrediction', axis=1)
low_risk_patients = low_risk_patients.drop('AtRiskPrediction', axis=1)

high_risk_stats = high_risk_patients.describe()

low_risk_stats = low_risk_patients.describe()

#print("High-Risk Patient Statistics:\n", high_risk_stats, "\n")
#print("Low-Risk Patient Statistics:\n", low_risk_stats)

mean_diff = abs(high_risk_stats.loc['mean'] - low_risk_stats.loc['mean'])
significant_features = mean_diff.sort_values(ascending=False)

#top 10 most significant features by difference in means
print("Top 10 Significant Features by Mean Difference:")
print(significant_features.head(10))

#a bit rudimentary
binary_features = high_risk_stats.columns[(high_risk_stats.loc['max'] <= 1.01) & (low_risk_stats.loc['max'] <= 1.01)]

binary_diff = abs(high_risk_stats.loc['mean', binary_features] - low_risk_stats.loc['mean', binary_features])

significant_binary_features = binary_diff.sort_values(ascending=False)

print("\nMost Significant Binary Features by Proportion Difference:")
print(significant_binary_features.head(10))

least_significant_features = mean_diff.sort_values(ascending=True)

print("Top 10 Least Significant Features by Mean Difference:")
print(least_significant_features.head(10))

least_significant_binary_features = binary_diff.sort_values(ascending=True)

print("\nLeast Significant Binary Features by Proportion Difference:")
print(least_significant_binary_features.head(10))

top_features = significant_features.sort_values(ascending=False).head(15).index.tolist()

cluster_means_significant = data_with_clusters_and_predictions.groupby('Cluster')[top_features].mean()

plt.figure(figsize=(10, 8))
sns.heatmap(cluster_means_significant, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Mean Significant Feature Values by Cluster")
plt.show()

top_binary_features = significant_binary_features.sort_values(ascending=False).head(15).index.tolist()

cluster_means_significant = data_with_clusters_and_predictions.groupby('Cluster')[top_binary_features].mean()

plt.figure(figsize=(10, 8))
sns.heatmap(cluster_means_significant, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Mean Significant Binary Feature Values by Cluster")
plt.show()

