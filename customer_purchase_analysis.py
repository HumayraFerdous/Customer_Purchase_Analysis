import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visuals
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Load the dataset (sample data)
data = {
    "CustomerID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Age": [25, 32, 45, 28, 50, 22, 38, 29, 42, 35],
    "Gender": ["M", "F", "M", "F", "M", "F", "F", "M", "F", "M"],
    "PurchaseAmount": [120, 85, 200, 60, 350, 45, 90, 110, 150, 180],
    "PaymentMethod": ["Credit", "Cash", "Credit", "Debit", "Credit", "Cash", "Debit", "Credit", "Debit", "Cash"],
    "SatisfactionRating": [4, 3, 5, 2, 5, 1, 3, 4, 4, 3]
}

df = pd.DataFrame(data)
print(df.head())

#print(df.isnull().sum())
numerical_cols = ['Age','PurchaseAmount','SatisfactionRating']
plt.figure(figsize=(12,4))
for i,col in enumerate(numerical_cols,1):
    plt.subplot(1,3,i)
    sns.boxplot(y=df[col])
plt.tight_layout()
#plt.show()

# Capping outliers
def cap_outliers(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column]<lower_bound,lower_bound,
                          np.where(df[column]>upper_bound,upper_bound,df[column]))
    return df
for col in numerical_cols:
    df = cap_outliers(df,col)

for i,col in enumerate(numerical_cols,1):
    plt.subplot(1,3,i)
    sns.boxplot(y=df[col])
plt.tight_layout()
#plt.show()

#Exploratory data analysis
#print(df.describe())

plt.figure(figsize=(8,6))
sns.heatmap(df[numerical_cols].corr(),annot = True,cmap='coolwarm')
plt.title("Correlation Matrix")
#plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df['PurchaseAmount'],bins = 10,kde = True)
plt.title("Distribution of Purchase Amount")
plt.xlabel("Purchase Amount ($)")
plt.ylabel("Frequency")
#plt.show()

#Average purchase by Gender
plt.figure(figsize=(8,5))
sns.barplot(x = "Gender", y ="PurchaseAmount",data = df, errorbar = None)
plt.title("Average Purchase Amount by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Purchase ($)")
#plt.show()

# Payment Method Preferences
plt.figure(figsize = (8,5))
sns.countplot(x='PaymentMethod',data=df)
plt.title("Payment Method Distribution")
plt.ylabel("Count")
#plt.show()

#Age vs PurChase Amount
plt.figure(figsize=(8,5))
sns.scatterplot(x = 'Age', y = 'PurchaseAmount',hue = "Gender",data = df, s = 100)
plt.title("Age vs. Purchase Amount")
plt.xlabel("Age")
plt.ylabel("Purchase Amount ($)")
#plt.show()



#Average Payment method by Age
plt.figure(figsize=(8,5))
sns.barplot(x="PaymentMethod",y="Age",data=df,errorbar = None)
plt.title("Average Payment Method by Age")
plt.xlabel("PaymentMethod")
plt.ylabel("Age")
#plt.show()

#SatisfactionRating vs PurchaseAmount
plt.figure(figsize=(8,5))
sns.scatterplot(x = 'SatisfactionRating', y = 'PurchaseAmount',hue = "Age",data = df, s = 100)
plt.title("SatisfactionRating vs. Purchase Amount")
plt.xlabel("SatisfactionRating")
plt.ylabel("Purchase Amount ($)")
plt.show()

#correlation between purchaseAmount and PaymentMethod
payment_method = {'Credit':0,'Case':1,'Debit':2}
df['pmethod'] = df['PaymentMethod'].map(payment_method)
print(df[['PurchaseAmount', 'pmethod']].corr())