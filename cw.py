#use version pythin vesrsion 3.9 below
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import feature_selection, metrics
from sklearn.preprocessing import StandardScaler

#load csv
#rawData = pd.read_csv("diabetic_data.csv")
#Changed file directory - Best
rawData = pd.read_csv("CW\diabetic_data.csv")

#show shape
#print(rawData.shape)
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
#print(rawData.shape)


#age transformation method
def get_middle_age(age_range):
    start, end = age_range[1:-1].split("-")  # Remove brackets and split by '-'
    middle_age = (int(start) + int(end)) / 2
    return middle_age

#applying to data
rawData['age'] = rawData['age'].apply(get_middle_age)
#print(rawData.head())


#diag selection and replacing all missing values with 0
columns_to_replace = ['diag_1', 'diag_2', 'diag_3']
rawData[columns_to_replace] = rawData[columns_to_replace].fillna(0)


#then drop any rows with missing values
rawData.dropna(inplace=True)
#print(rawData.shape)
#print(rawData.dtypes)


#numerical columns
numerical_columns = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses',
    'age', 'patient_nbr',
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'diag_1', 'diag_2', 'diag_3', 'encounter_id'
]


#categorical columns
categorical_columns = [
    'race', 'gender', 'payer_code', 'medical_specialty',
    'max_glu_serum', 'A1Cresult', 
    'metformin',  'patient_nbr',
    'glimepiride',  'glipizide', 'glyburide', 
    'pioglitazone', 'rosiglitazone',  
    'insulin',
    'change', 'diabetesMed', 'readmitted'
]


#excluding useless cols
exclude_columns = ['payer_code']


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
filtered_data = filtered_data.drop_duplicates(subset=['patient_nbr'], keep='first')


print(filtered_data.shape)
#print(filteredCategorical_data.shape)


filtered_data.to_csv('filtered_data.csv', index=False)


########################################PART 2############################################


#BAR CHART FOR FREQUENCY OF READMISSION BY AGE
filtered_data['readmitted_binary'] = (filtered_data['readmitted'] != 'NO').astype(int) #important line for rest of code

age_readmission_frequency = filtered_data.groupby('age')['readmitted_binary'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(age_readmission_frequency['age'], age_readmission_frequency['readmitted_binary'], color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency of Readmission')
plt.title('Frequency of Readmission by Age')
plt.xticks(rotation=45)
plt.tight_layout()  
#plt.show()

#BAR CHART FOR FREQUENCY OF READMISSION BY RACE - adjusted for demographics too
race_readmission_frequency = filtered_data.groupby('race')['readmitted_binary'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(race_readmission_frequency['race'], race_readmission_frequency['readmitted_binary'], color='skyblue')
plt.xlabel('Race')
plt.ylabel('Frequency of Readmission')
plt.title('Frequency of Readmission by Race')
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
plt.ylabel('Frequency of Readmission')
plt.title('Frequency of Readmission by Gender')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()

#BAR CHART FOR DIAGS
diag_cols = ['diag_1', 'diag_2', 'diag_3']
melted_data = pd.melt(filtered_data, id_vars=['readmitted_binary'], value_vars=diag_cols, var_name='Diagnosis', value_name='Code')
diag_readmission_frequency = melted_data.groupby('Code')['readmitted_binary'].sum().reset_index().sort_values(by='readmitted_binary', ascending=False)
#FOR ALL DIAGS AS NUMBERS
plt.figure(figsize=(12, 8))
plt.bar(diag_readmission_frequency['Code'], diag_readmission_frequency['readmitted_binary'], color='skyblue')
plt.xlabel('Diagnosis Code')
plt.ylabel('Frequency of Readmission')
plt.title('Diagnosis Codes by Frequency of Readmission')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()
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

def map_diag_to_category(diag_code):
    try:
        diag_code_float = float(diag_code) #in case of decimals
        diag_code_int = int(diag_code_float)
        for category, icd_range in categories.items():
            if diag_code_int in icd_range:
                return category
    except ValueError:
        # If diag_code cannot be converted to float, it might be a special code or missing; categorize as 'Other'
        return 'Other'
    return 'Other'

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
model_df['readmitted_chance'] = readmitted_chance
model_df.to_csv('model_df.csv', index=False)

