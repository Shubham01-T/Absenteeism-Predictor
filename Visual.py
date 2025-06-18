import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
df = pd.read_csv('Absenteeism_Predictions.csv')

# Set a nice Seaborn style
sns.set_theme(style="whitegrid", palette="colorblind")

# 1. Heatmap of Correlations
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(" Correlation Heatmap")
plt.tight_layout()
plt.show()

#  Probability vs Age (Colored by Prediction)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Probability', hue='Prediction', palette='Set1')
plt.title('Probability of Absenteeism by Age (Colored by Prediction)')
plt.xlabel("Age")
plt.ylabel("Probability")
plt.legend(title='Prediction')
plt.tight_layout()
plt.show()



# Count Plot of Reasons for Absence
plt.figure(figsize=(12, 6))
reason_cols = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
reason_sums = df[reason_cols].sum().reset_index()
reason_sums.columns = ['Reason', 'Count']
sns.barplot(data=reason_sums, x='Reason', y='Count', palette='magma')
plt.title(" Distribution of Absence Reasons")
plt.xlabel("Reason")
plt.ylabel("Number of Times Selected")
plt.tight_layout()
plt.show()


