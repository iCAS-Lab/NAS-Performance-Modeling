import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import joblib


data = pd.read_csv('./data/dataset.csv')
real_data = pd.read_csv('./data/real_data/real_models.csv').dropna()

X = data[['Total Parameters', 'Total MACs'] + [f'K{k}x{k} MAC Percentage' for k in range(1,8)]]
y = data[['GPU_energy_per_inference', 'average_latency(ms)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [750], 
    'max_depth': [30], 
    'min_samples_split': [20],
}
rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

y_pred = grid_search.best_estimator_.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_percentage_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')
print(f'Mean Squared Error (MSE): GPU Energy: {mse[0]}, Latency: {mse[1]}')
print(f'Mean Absolute Percent Error: GPU Energy: {mae[0]}, Latency: {mae[1]}')
print(f'R^2 Score: GPU Energy: {r2[0]}, Latency: {r2[1]}')

for index, row in real_data.iterrows():
    model_data = {
        'Total Parameters': [row['Total Parameters']], 
        'Total MACs': [row['Total MACs']]
    }
    for k in range(1,8):
        model_data[f'K{k}x{k} MAC Percentage'] = [row[f'K{k}x{k} MAC Percentage']]
    
    observed_data = {
        'GPU_energy_per_inference': row['GPU_energy_per_inference'],
        'average_latency(ms)': row['average_latency(ms)']
    }
    
    new_model_df = pd.DataFrame(model_data)
    
    predicted_values = grid_search.best_estimator_.predict(new_model_df)
    print(predicted_values[0])
    
    print('---------- Error for ' + row['Model Name'] + ' ----------')
    for index, label in enumerate(observed_data):
        pred = predicted_values[0][index]
        per_error = mean_absolute_percentage_error([observed_data[label]], [pred], multioutput='raw_values')[0]
        print(f"{label}: {per_error}")
    print('')

joblib.dump(grid_search.best_estimator_, 'random_forest.joblib')
print("GridSearchCV model saved!")

best_model = grid_search.best_estimator_
feature_importances = best_model.feature_importances_

# Compute Permutation Importance on validation or test set
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Perm Importance': perm_importance.importances_mean
}).sort_values(by='Perm Importance', ascending=False)

features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

import numpy as np

# Create a combined DataFrame for both Gini and Permutation Importances
combined_importance_df = pd.DataFrame({
    'Feature': features,
    'Gini Importance': feature_importances,
    'Perm Importance': perm_importance.importances_mean
}).sort_values(by='Gini Importance', ascending=False)
print("Importance: \n", combined_importance_df)
# Plot both Gini and Permutation Importances side by side
plt.figure(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(combined_importance_df))
plt.barh(index, combined_importance_df['Gini Importance'], bar_width, label='Gini Importance', color='blue')
plt.barh(index + bar_width, combined_importance_df['Perm Importance'], bar_width, label='Permutation Importance', color='red')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Gini Importance vs Permutation Importance')
plt.yticks(index + bar_width / 2, combined_importance_df['Feature'])
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
#plt.show()