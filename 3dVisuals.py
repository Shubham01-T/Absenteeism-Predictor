import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('Absenteeism_Predictions.csv')

# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x='Age',
    y='Transportation Expense',
    z='Probability',
    color='Prediction',
    title=' 3D Scatter Plot: Age vs Transport Expense vs Absenteeism Probability',
    labels={
        'Age': 'Age',
        'Transportation Expense': 'Transport Expense',
        'Probability': 'Absenteeism Probability',
        'Prediction': 'Prediction'
    },
    opacity=0.7
)

# Show the interactive plot
fig.show()
