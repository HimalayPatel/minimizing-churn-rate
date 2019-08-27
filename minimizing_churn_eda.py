import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("C:\\Users\\himal\\Desktop\\Machine Learning Practicals\\P4- Minimizing Churn Rate through Analysis of Financial Habits\\P39-Minimizing-Churn-Data\\churn_data.csv") # Users who were 60 days enrolled, churn in the next 30

# EDA
# print(dataset.head(5))
# print(dataset.columns)
# print(dataset.describe())
# print(dataset.isna().any())
# print(dataset.isna().sum())
dataset=dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])

# Histograms
dataset2 = dataset.drop(columns=['user', 'churn','housing','payment_type','zodiac_sign'])
fig = plt.figure(figsize=(20, 15))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 4, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#FF0077')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Pie Charts
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(5, 4, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
    values = dataset2.iloc[:, i - 1].value_counts(normalize=True).values
    index = dataset2.iloc[:, i - 1].value_counts(normalize=True).index
    plt.pie(values, labels=index, autopct='%1.1f%%')
    plt.axis('equal')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
# Exploring Uneven Features
print(dataset[dataset2.waiting_4_loan == 1].churn.value_counts())
print(dataset[dataset2.cancelled_loan == 1].churn.value_counts())
print(dataset[dataset2.received_loan == 1].churn.value_counts())
print(dataset[dataset2.rejected_loan == 1].churn.value_counts())
print(dataset[dataset2.left_for_one_month == 1].churn.value_counts())

# Correlation with the Response Variable
dataset.drop(columns=['user', 'churn', 'housing', 'payment_type',
                      'registered_phones', 'zodiac_sign']).corrwith(dataset.churn).plot.bar(
    figsize=(20,10),title = 'Correlation with Response variable',
              fontsize = 15, rot = 45,
              grid = True)
plt.tight_layout()
plt.show()

# Correlation Matrix
sns.set(style="white")
corr = dataset.drop(columns = ['user', 'churn','housing','payment_type','zodiac_sign']).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.suptitle("Correlation Matrix",fontsize=20)
plt.show()

dataset = dataset.drop(columns = ['app_web_user'])
# dataset = dataset.drop(columns = ['housing','payment_type','zodiac_sign'])
# Note: Although there are somewhat correlated fields, they are not colinear
# These feature are not functions of each other, so they won't break the model
# But these feature won't help much either. Feature Selection should remove them.

dataset.to_csv('new_churn_data.csv', index = False)

