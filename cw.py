#use version pythin vesrsion 3.9 below
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import feature_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.optimizers import Adam
from keras_tuner import Objective
import json




import shap
import os

print(tf.__version__)

#hide gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#restore
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Assuming you want to use the first GPU

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


#avoid data leakage by just keeping one
rawData = rawData.drop_duplicates(subset='patient_nbr', keep='first')


#diag selection and replacing all missing values with 0
columns_to_replace = ['diag_1', 'diag_2', 'diag_3']
rawData[columns_to_replace] = rawData[columns_to_replace].fillna(0)

#mgs and ac1 -ACTUALLY, the abscense of such tests might be risky towards readmission
#rawData['max_glu_serum'] = rawData['max_glu_serum'].replace('None', np.nan, inplace=True) 
#rawData['AC1Result'] = rawData['AC1Result'].replace('None', np.nan, inplace=True)

#### WE ARE CURRENTLY DROPPING ALOT OF DATA AS BOTH 'PAYER_CODE' AND 'MEDICAL_SPECIALITY' HAS ALOT OF EMPTY DATA ####
#### TO BE PRECISE: payer_code: 39.56% AND medical_specialty: 49.08% is empty ####
#### DATA IS DROPPED FROM (101766, 33) to (27140, 33) ####

#some tuning to reduce overfitting for tensrflow model  MAYBE using insights from xgboost feature importance
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

age_readmission_frequency = filtered_data.groupby('age')['readmitted_binary'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(age_readmission_frequency['age'], age_readmission_frequency['readmitted_binary'], color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency of Readmission within 30d')
plt.title('Frequency of Readmission within 30d by Age')
plt.xticks(rotation=45)
plt.tight_layout()  
#plt.show()

#BAR CHART FOR FREQUENCY OF READMISSION BY RACE - adjusted for demographics too
race_readmission_frequency = filtered_data.groupby('race')['readmitted_binary'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(race_readmission_frequency['race'], race_readmission_frequency['readmitted_binary'], color='skyblue')
plt.xlabel('Race')
plt.ylabel('Frequency of Readmission within 30d')
plt.title('Frequency of Readmission within 30d by Race')
plt.xticks(rotation=45)
plt.tight_layout()  
#plt.show()

## https://www.visualcapitalist.com/visualizing-u-s-population-by-race/
#CAUCASIAN 60.1%
#HISPANIC 18.5%
#BLACK 12.2%
#ASIAN 5.6% 
#Our chart does not follow this pattern, african americans are disproportionately affected by diabetes in the US

#BAR CHART FOR FREQUENCY OF READMISSION BY GENDER
gender_readmission_frequency = filtered_data.groupby('gender')['readmitted_binary'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(gender_readmission_frequency['gender'], gender_readmission_frequency['readmitted_binary'], color='skyblue')
plt.xlabel('Gender')
plt.ylabel('Frequency of Readmission within 30d')
plt.title('Frequency of Readmission within 30d by Gender')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()

#BAR CHART FOR DIAGS
diag_cols = ['diag_1', 'diag_2', 'diag_3']
melted_data = pd.melt(filtered_data, id_vars=['readmitted_binary'], value_vars=diag_cols, var_name='Diagnosis', value_name='Code')
diag_readmission_frequency = melted_data.groupby('Code')['readmitted_binary'].sum().reset_index().sort_values(by='readmitted_binary', ascending=False)
#WE NEED 3 BAR CHARTS - ONE FOR DIAG1 CONVERTED TO DISEASE NAME - THEN FOR 2 AND 3

categories = {
    'Certain infectious and parasitic diseases': range(1, 140),
    'Neoplasms': range(140, 240),
    'Diseases of the blood and blood-forming organs': range(280, 290),
    'Endocrine, nutritional and metabolic diseases': range(240, 280),
    'Mental and behavioural disorders': range(290, 320),
    'Diseases of the nervous system': range(320, 360),
    'Diseases of the eye and adnexa': range(360, 390),
    'Diseases of the ear and mastoid process': range(390, 460),
    'Diseases of the circulatory system': range(390, 460),
    'Diseases of the respiratory system': range(460, 520),
    'Diseases of the digestive system': range(520, 580),
    'Diseases of the skin and subcutaneous tissue': range(680, 710),
    'Diseases of the musculoskeletal system and connective tissue': range(710, 740),
    'Diseases of the genitourinary system': range(580, 630),
    'Pregnancy, childbirth and the puerperium': range(630, 680),
    'Certain conditions originating in the perinatal period': range(760, 780),
    'Congenital malformations, deformations and chromosomal abnormalities': range(740, 760),
    'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified': range(780, 800),
    'Injury, poisoning and certain other consequences of external causes': range(800, 1000),
    'External causes of morbidity and mortality': range(800, 1000),
    'Factors influencing health status and contact with health services': range(1000, 1100),
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
    'Problems with Special Senses and Functions, and Transplant Status': range(41, 50),  # V41-V49
    'Surgical Aftercare and Other Postprocedural States': range(50, 60),  # V50-V59
    'Health Services Encounters for Other Circumstances': range(60, 70),  # V60-V69
    'General Medical Examinations and Special Investigations': range(70, 80),  # V70-V79
    'Special Screenings for Diseases and Disorders': range(80, 92)  # V80-V91
}



def map_diag_to_category(diag_code):
    try:
        diag_code_float = float(diag_code) #in case of decimals
        diag_code_int = int(diag_code_float)
        for category, icd_range in categories.items():
            if diag_code_int in icd_range:
                return category
    except ValueError:
        if not diag_code.startswith('V'):
            return 'Other'
        # Extract the numeric part of the V-code and convert to an integer for comparison
        try:
            code_num = int(diag_code[1:])
            for category, code_range in v_code_categories.items():
                if code_num in code_range:
                    return category
        except ValueError:
            # Handle case where the conversion fails (should not happen with correct V-codes)
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


########################################PART 3############################################

model_columns = ['num_medications', 'number_outpatient', 'number_emergency', 'time_in_hospital', 'number_inpatient','encounter_id', 'age', 'num_lab_procedures', 'number_diagnoses','num_procedures', 'readmitted_binary']
model_df=filtered_data[model_columns]
#print(model_df.head(5))

#LINEAR MODEL USING ALL VARS
model = smf.ols(formula='readmitted_binary~num_medications+age+number_outpatient+number_emergency+time_in_hospital+number_inpatient+encounter_id+num_lab_procedures+number_diagnoses+num_procedures', data=model_df).fit()
#print('Model1 coefs are {}'.format(model.params))
#print(model.summary())

readmitted_chance=model.predict(model_df)
model_df.loc[:, 'readmitted_chance'] = readmitted_chance
model_df.to_csv('model_df.csv', index=False) #OPEN FILE IN EXCEL. CLICK CONVERT!!! THEN SORT BY READMITTED CHANCE - YOULL SEE THAT IT LOOKS LIKE IT WORKS!! MAKE SURE TO EXPAND SELECTION IF ASKED

###########################################################################################

#race_dummies = pd.get_dummies(filtered_data['race'], prefix='race')
#filtered_data = pd.concat([filtered_data, race_dummies], axis=1)
#filtered_data = filtered_data.drop('race', axis=1)
print(filtered_data.dtypes)

##some feature engineering, assumption is that if one has multiple diagnoses then they have a higher chance of being prone to illness in general. 
##Could possibly weight it? So Diag1=1 + Diag2=1.5 + Diag3=1.75
##Even deeper. Use the weights of the categories themselves then normalise. 
#filtered_data['total_severity'] = filtered_data[['diag_1', 'diag_2', 'diag_3']].applymap(lambda x: diag_severity.get(x, 0)).sum(axis=1)
filtered_data['multiple_diagnoses'] = filtered_data[['diag_1', 'diag_2', 'diag_3']].notnull().sum(axis=1) > 1
filtered_data['multiple_diagnoses'] = filtered_data['multiple_diagnoses'].astype(int)

##More feature engineering: Count the number of medications a patient is on. This could indicate the severity of their diabetes management.
medications = ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
filtered_data['diabetes_medication_count'] = filtered_data[medications].apply(lambda x: x != 'No').sum(axis=1)

#filtered_data['age_gender_interaction'] = filtered_data['age'] * filtered_data['gender']

filtered_data.to_csv('filtered_data.csv', index=False)

def normalise(df):
    num_cols=df.select_dtypes(include=[np.number]).copy()
    df_norm=((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    return df_norm 


#these things have nothing to do with our data anymore. diags have been categorised and the rest are just useless
filtered_data =filtered_data.drop('readmitted', axis=1)
filtered_data =filtered_data.drop('encounter_id', axis=1)
filtered_data =filtered_data.drop('patient_nbr', axis=1)
filtered_data =filtered_data.drop('diag_1', axis=1) #since weve categorised them
filtered_data =filtered_data.drop('diag_2', axis=1)
filtered_data =filtered_data.drop('diag_3', axis=1)

#data leakage as this has a direct connection to patient status post-visit, which could imply information about readmission.
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

print(X_train_smote.shape)

classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_smote)
class_weightsB4 = compute_class_weight('balanced', classes=classes, y=y_train)
print("Class weights before smote: ", class_weightsB4)
print("Class weights after smote: ", class_weights)

#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
# print(X_train.dtypes)
# #clf.fit(X_train, y_train)
# clf.fit(X_train_smote, y_train_smote)
# predictions = clf.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print('\n ############################################### \n GRADIENT BOOST \n')
# print("Accuracy:", accuracy)
# precision = precision_score(y_test, predictions)
# recall = recall_score(y_test, predictions)
# f1 = f1_score(y_test, predictions)
# roc_auc = roc_auc_score(y_test, predictions)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("ROC AUC Score:", roc_auc)
# conf_matrix = confusion_matrix(y_test, predictions)
# print("Confusion Matrix:\n", conf_matrix)

# print('\n ############################################### \n RANDOM FOREST \n')

# #rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # 100 trees in the forest
# rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42, class_weight='balanced')
# #rf_classifier.fit(X_train, y_train)
# rf_classifier.fit(X_train_smote, y_train_smote)
# y_pred = rf_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
# print("ROC AUC Score:", roc_auc)
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", conf_matrix)

print('\n ############################################### \n XGBOOST \n')

xgb_classifier = XGBClassifier(
    n_estimators=1000,  #number of trees
    max_depth=6,  #depth of each tree
    learning_rate=0.01,  #slower, more robust learning
    subsample=0.8,  #subset data to prevent overfitting
    colsample_bytree=0.8,  #subset of features to prevent overfitting
    reg_lambda=1.0,  #L2 regularization term
    reg_alpha=0.0,  #L1 regularization term
    tree_method='hist',  #histogram-based tree method for faster computation
    device='cuda',  #GPU acceleration
    random_state=42
)
xgb_classifier.fit(X_train_smote, y_train_smote)
y_pred = xgb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
importances = xgb_classifier.feature_importances_
for i, importance in enumerate(importances):
    print(f'Feature: {X_train.columns[i]}, Score: {importance}')

plt.figure(figsize=(10, 8))
plt.barh(range(len(importances)), importances, tick_label=X_train.columns)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance Scores')
#plt.show()

print('\n ############################################### \n TENSORFLOW \n')

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_precision',
    mode='max',
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Define ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    filepath='/models/model.h5',
    monitor='val_loss',
    save_best_only=True  # Only save a model if `val_loss` has improved
)
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

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
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

model.summary()

#model.fit(X_train_scaled, y_train_smote, epochs=10, batch_size=64, validation_split=0.2) #batch size 128 or 64 is good


#results = model.evaluate(X_test_scaled, y_test, verbose=1)


##VALIDATE#####################################################################

# def build_model(hp):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Dense(hp.Int('units_input', min_value=32, max_value=512, step=32),
#                                     activation='relu',
#                                     input_shape=(X_train_scaled.shape[1],),
#                                     kernel_regularizer=l2(hp.Float('l2_input', min_value=0.0001, max_value=0.01, sampling='LOG'))))
#     model.add(tf.keras.layers.Dropout(hp.Float('dropout_input', min_value=0.0, max_value=0.5, step=0.1)))

#     for i in range(6):  # Adjust the range as per your model's structure
#         model.add(tf.keras.layers.Dense(hp.Int(f'units_layer_{i}', min_value=32, max_value=512, step=32),
#                                         activation='relu',
#                                         kernel_regularizer=l2(hp.Float(f'l2_layer_{i}', min_value=0.0001, max_value=0.01, sampling='LOG'))))
#         model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.1)))
#         model.add(tf.keras.layers.BatchNormalization())

#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
#     model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc'), F1Score()])
#     return model



# ##FIND BEST HYPERPARAMETERS

# tuner = BayesianOptimization(
#      build_model,
#      objective=Objective("val_accuracy", direction="max"),
#      max_trials=20,
#      executions_per_trial=2,
#      directory='my_dir',
#     project_name='bayesian_optim'
# )

# # tuner.search(X_train_scaled, y_train_smote, epochs=10, validation_split=0.2)
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]


# print(best_hps)
# model = build_model(best_hps)



##THEN DO KFOLD


num_folds = 100
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
validation_losses = []
validation_accuracies = []
validation_data_per_fold = []

if not os.path.exists('models'):
    os.makedirs('models')


for fold, (train_indices, val_indices) in enumerate(kfold.split(X_train_scaled, y_train_smote)):
    print(f'Fold {fold+1}/{num_folds}:')


    checkpoint_filepath = f'models/best_model_fold_{fold+1}.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    

    # Get the training and validation data for this fold
    X_train_fold, X_val_fold = X_train_scaled[train_indices], X_train_scaled[val_indices]
    y_train_fold, y_val_fold = y_train_smote[train_indices], y_train_smote[val_indices]

    

    # Train the model on the training data for this fold
    history = model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=64, validation_data=(X_val_fold, y_val_fold), verbose=0, callbacks=[model_checkpoint_callback])
    #for bayesian opt DONT DO THIS HAHAH
    #history = model.fit(X_train_fold, y_train_fold, epochs=best_hps.get('epochs'), batch_size=best_hps.get('batch_size'), validation_split=0.2)
    validation_data_per_fold.append((X_val_fold, y_val_fold))


    # Evaluate the model on the validation data for this fold
    eval_results = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    loss = eval_results[0]  # Extract loss
    accuracy = eval_results[1]  # Extract accuracy
    precision = eval_results[2]
    recall = eval_results[3]
    auc= eval_results[4]
    f1 = eval_results[5]

    train_eval_results = model.evaluate(X_train_fold, y_train_fold, verbose=0)

    print(f'Training loss for this fold: {train_eval_results[0]}, accuracy: {train_eval_results[1]}, precision: {train_eval_results[2]}, recall: {train_eval_results[3]}, AUC: {train_eval_results[4]}, F1 score: {train_eval_results[5]}')

    print(f'Validation loss for this fold: {loss}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, AUC: {auc}, F1 score: {f1} \n')




    # Store the evaluation results for this fold
    validation_losses.append(loss)
    validation_accuracies.append(accuracy)

mean_validation_loss = np.mean(validation_losses)
mean_validation_accuracy = np.mean(validation_accuracies)
std_validation_loss = np.std(validation_losses)
std_validation_accuracy = np.std(validation_accuracies)

# Print the mean and standard deviation of validation metrics
print(f'\nMean validation loss: {mean_validation_loss} (±{std_validation_loss})')
print(f'\nMean validation accuracy: {mean_validation_accuracy} (±{std_validation_accuracy})')





# explainer = shap.DeepExplainer(model, X_train_scaled[:100])  # Using a subset as the background dataset for efficiency
# shap_values = explainer.shap_values(X_test_scaled[:100])  # Using a subset for demonstration
# print(type(shap_values))
# print(np.array(shap_values).shape)
# print(explainer.expected_value)
# shap.initjs()  # Initializes JavaScript visualization in the notebook
# shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test_scaled[0])
# shap.summary_plot(shap_values[1], X_test_scaled[:100])
# shap.dependence_plot("age", shap_values[1], X_test_scaled[:100])






# Assuming the first value is loss and subsequent values are the metrics in the order added
# test_loss = results[0]
# test_acc = results[1]
# test_precision = results[2]
# test_recall = results[3]
# test_auc = results[4]

# print(f"Loss: {test_loss}, Accuracy: {test_acc}, Precision: {test_precision}, Recall: {test_recall}, AUC: {test_auc}")
