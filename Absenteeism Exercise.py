from absenteeism_module import *
# from load_and_clean_data import load_and_clean_data  # No need to import separately

model = absenteeism_model('model', 'scaler')

# Call the load_and_clean_data method using the model instance
model.load_and_clean_data('Absenteeism_new_data.csv')

# Get the predicted outputs
model.predicted_outputs()

# Save the predicted outputs to a CSV file
model.predicted_outputs().to_csv('Absenteeism_Predictions.csv', index=False)
