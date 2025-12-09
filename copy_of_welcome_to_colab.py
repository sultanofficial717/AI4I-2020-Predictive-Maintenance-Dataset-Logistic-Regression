

import pandas as pd
df=pd.read_csv('/content/ai4i2020.csv')
print(df.head(10))

import pandas as pd
# Ensure df is available, if not, this will raise an error. Assuming df is already loaded.
# Based on the notebook state, df is loaded in cell hGXuKr7YLI9q.

# Define df_dropped before it is used
df_dropped=df.drop(['UDI','Product ID','Type'],axis=1)

columns_to_count = ['HDF', 'PWF', 'OSF', 'RNF']

for col in columns_to_count:
    print(f"\nCounts for column '{col}':")
    counts = df_dropped[col].value_counts().sort_index()
    print(counts)

"""### Initial Data Inspection and Column Dropping

After loading the dataset, we inspect its initial rows and then proceed to drop columns that are not relevant for the current analysis ('TWF', 'HDF', 'PWF', 'OSF', 'RNF').
"""

df_dropped=df.drop(['UDI','Product ID','Type'],axis=1)

print(df_dropped.head(10))

"""### DataFrame Information

This cell provides a concise summary of the `df_dropped` DataFrame, including the data types of each column, the number of non-null values, and memory usage. This helps in understanding the structure and completeness of the data.
"""

print("Data Set Info:")
df_dropped.info()

"""### Outlier Detection for 'Air temperature [K]'

This section visualizes the distribution of 'Air temperature [K]' using a box plot and performs statistical outlier detection using the Interquartile Range (IQR) method. Outliers are identified as data points falling below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`.
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Air temperature [K]'], color='green')
plt.xlabel('Air temperature [K]')
plt.title('Box Plot: Air Temperature Outlier Detection')
plt.show()

# Statistical outlier detection
Q1 = df['Air temperature [K]'].quantile(0.25)
Q3 = df['Air temperature [K]'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Air temperature [K]'] < lower_bound) |
              (df['Air temperature [K]'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}K")
print(f"Upper bound: {upper_bound:.2f}K")

"""### Outlier Detection for 'Process temperature [K]'

Similar to air temperature, this section analyzes 'Process temperature [K]' for outliers using a box plot and the IQR method. It helps in identifying unusually high or low process temperature readings.
"""

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Process temperature [K]'], color='yellow')
plt.xlabel('Process temperature [K]')
plt.title('Box Plot: Process Temperature Outlier Detection')
plt.show()

# Statistical outlier detection
Q1 = df['Process temperature [K]'].quantile(0.25)
Q3 = df['Process temperature [K]'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Process temperature [K]'] < lower_bound) |
              (df['Process temperature [K]'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}K")
print(f"Upper bound: {upper_bound:.2f}K")

"""### Outlier Detection for 'Rotational speed [rpm]'

This section focuses on detecting outliers in 'Rotational speed [rpm]'. A box plot provides a visual summary, and the IQR method quantitatively identifies extreme values, which are then categorized as low or high outliers.
"""

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Rotational speed [rpm]'], color='blue')
plt.xlabel('Rotational speed [rpm]')
plt.title('Box Plot: Rotational speed [rpm] Outlier Detection')
plt.show()

# Statistical outlier detection
Q1 = df['Rotational speed [rpm]'].quantile(0.25)
Q3 = df['Rotational speed [rpm]'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Rotational speed [rpm]'] < lower_bound) |
              (df['Rotational speed [rpm]'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}K")
print(f"Upper bound: {upper_bound:.2f}K")

"""### Categorization of Rotational Speed Outliers

Following the outlier detection for 'Rotational speed [rpm]', this section categorizes each data point as an 'inliner', 'low_outlier', or 'high_outlier' based on the previously calculated IQR bounds. It also prints the count of each outlier type.
"""

import seaborn as sns
from scipy.stats import pearsonr

feature_name='Rotational speed [rpm]'
target_name='Machine failure'

Q1=df_dropped[feature_name].quantile(0.25)
Q3=df_dropped[feature_name].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

df_dropped['is_outlier']=(df_dropped[feature_name]<lower_bound)|(df_dropped[feature_name]>upper_bound)
df_dropped['outlier_type']= 'inliner'
df_dropped.loc[df_dropped[feature_name]<lower_bound, 'outlier_type']= 'low_outlier'
df_dropped.loc[df_dropped[feature_name]>upper_bound, 'outlier_type']= 'high_outlier'

print(f"Found {df_dropped['is_outlier'].sum()}outlier:")
print(f"- Low outlier: {(df_dropped['outlier_type']=='low_outlier').sum()}")
print(f"- High outlier: {(df_dropped['outlier_type']=='high_outlier').sum()}")

"""### Correlation Analysis of Rotational Speed Outliers

This heatmap visualizes the correlation matrix of numerical features specifically for the data points identified as outliers in 'Rotational speed [rpm]'. It helps understand how these outlier conditions might be related to other operational parameters, including 'Machine failure'.
"""

numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']
outliers_numerical = outliers[numerical_cols]

correlation_matrix = outliers_numerical.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Outliers based on Rotational Speed')
plt.show()

"""### Correlation Analysis of Numerical Features

This heatmap visualizes the Pearson correlation coefficients between all numerical features in the `df_dropped` DataFrame. The values range from -1 to 1, where:
-   **1** indicates a perfect positive linear correlation.
-   **-1** indicates a perfect negative linear correlation.
-   **0** indicates no linear correlation.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Select only numerical columns for correlation analysis from df_dropped
numerical_features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Machine failure'
]

correlation_matrix_full = df_dropped[numerical_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_full, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features in df_dropped')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Torque [Nm]'], color='brown')
plt.xlabel('Torque [Nm]')
plt.title('Box Plot: Rotational speed [rpm] Outlier Detection')
plt.show()

# Statistical outlier detection
Q1 = df['Torque [Nm]'].quantile(0.25)
Q3 = df['Torque [Nm]'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Torque [Nm]'] < lower_bound) |
              (df['Torque [Nm]'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}K")
print(f"Upper bound: {upper_bound:.2f}K")

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Tool wear [min]'], color='gray')
plt.xlabel('Tool wear [min]')
plt.title('Box Plot: Rotational speed [rpm] Outlier Detection')
plt.show()

# Statistical outlier detection
Q1 = df['Tool wear [min]'].quantile(0.25)
Q3 = df['Tool wear [min]'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Tool wear [min]'] < lower_bound) |
              (df['Tool wear [min]'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f}K")
print(f"Upper bound: {upper_bound:.2f}K")

"""**Test Train Split**"""

from sklearn.model_selection import train_test_split
import pandas as pd

# Exclude the target variable and temporary/non-numeri|cal columns that caused scaler errors
x = df_dropped.drop(['Machine failure', 'is_outlier', 'outlier_type'], axis=1)
y = df_dropped['Machine failure']

x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"X_train shape: {x_train.shape[0]}")
print(f"X_test shape: {X_test.shape[0]}")

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)

"""# **Feature Scaling**"""

df_dropped.describe()

df_dropped.hist(figsize=(12,10),bins=30)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=x_train.columns, index=x_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=x_test.columns, index=x_test.index)

X_train_scaled.hist(figsize=(12,10),bins=30)
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model=LogisticRegression()
model.fit(X_train_scaled,y_train)

y_pred=model.predict(X_test_scaled)

print("Accuracy",accuracy_score(y_test,y_pred))
print("\nClassification", classification_report(y_test,y_pred))
print



from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the positive class
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()