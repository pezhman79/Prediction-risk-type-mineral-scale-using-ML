"""
Created on Mon Sep  1 18:04:07 2025
Enhanced with Multiple Models Comparison
@author: pejma
"""

import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import shap
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# 1) Load & preprocess
# --------------------------
data = pd.read_excel(r"E:\AUT\msc project\scale\scale data\excel\Dataset-Scale.XLSX")
data = data.dropna().drop_duplicates().drop(columns=[])
data = data.drop(columns=['SiO2-Quartz'])

X = data.drop(columns=['CaSO4-Anhydrite', 'CaCO3-Aragonite', 'BaSO4-Barite', 'CaCO3-Calcite', 'SrSO4-Celestite',
                       'CaSO4:2H2O-Gypsum', 'NaCl-Halite', 'FeCO3-Siderite']) 
y_target = data[['CaSO4-Anhydrite', 'CaCO3-Aragonite', 'BaSO4-Barite', 'CaCO3-Calcite', 'SrSO4-Celestite',
                       'CaSO4:2H2O-Gypsum', 'NaCl-Halite', 'FeCO3-Siderite']]

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_target, test_size=0.30, random_state=48
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=48
)

# --------------------------
# 2) XGBoost Optimization
# --------------------------
print("=" * 60)
print("OPTIMIZING XGBOOST MODEL")
print("=" * 60)

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    xgb_params = {
        'reg__estimator__eta': trial.suggest_float('reg__estimator__eta', 0.01, 0.2),
        'reg__estimator__max_depth': trial.suggest_int('reg__estimator__max_depth', 3, 5),
        'reg__estimator__subsample': trial.suggest_float('reg__estimator__subsample', 0.6, 1.0),
        'reg__estimator__colsample_bytree': trial.suggest_float('reg__estimator__colsample_bytree', 0.6, 1.0),
        'reg__estimator__reg_lambda': trial.suggest_float('reg__estimator__reg_lambda', 1e-2, 100, log=True),
        'reg__estimator__reg_alpha': trial.suggest_float('reg__estimator__reg_alpha', 1e-2, 100, log=True),
        'reg__estimator__n_estimators': trial.suggest_int('reg__estimator__n_estimators', 200, 1200, step=100),
    }

    base_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('reg', MultiOutputRegressor(base_xgb))
    ])

    pipeline.set_params(**xgb_params)
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    return r2_val

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train, X_val, y_val),
                   n_trials=10, show_progress_bar=True)

print("\n=== XGBoost Best Parameters ===")
print(study_xgb.best_params)
print(f"Best Validation Score: {study_xgb.best_value:.4f}")

# --------------------------
# 3) Random Forest Optimization
# --------------------------
print("\n" + "=" * 60)
print("OPTIMIZING RANDOM FOREST MODEL")
print("=" * 60)

def objective_rf(trial, X_train, y_train, X_val, y_val):
    rf_params = {
        'reg__estimator__n_estimators': trial.suggest_int('reg__estimator__n_estimators', 100, 500, step=50),
        'reg__estimator__max_depth': trial.suggest_int('reg__estimator__max_depth', 5, 20),
        'reg__estimator__min_samples_split': trial.suggest_int('reg__estimator__min_samples_split', 2, 10),
        'reg__estimator__min_samples_leaf': trial.suggest_int('reg__estimator__min_samples_leaf', 1, 5),
        'reg__estimator__max_features': trial.suggest_categorical('reg__estimator__max_features', ['sqrt', 'log2', None]),
    }

    base_rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('reg', MultiOutputRegressor(base_rf))
    ])

    pipeline.set_params(**rf_params)
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    return r2_val

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, X_val, y_val),
                  n_trials=10, show_progress_bar=True)

print("\n=== Random Forest Best Parameters ===")
print(study_rf.best_params)
print(f"Best Validation Score: {study_rf.best_value:.4f}")

# --------------------------
# 4) Decision Tree Optimization
# --------------------------
print("\n" + "=" * 60)
print("OPTIMIZING DECISION TREE MODEL")
print("=" * 60)

def objective_dt(trial, X_train, y_train, X_val, y_val):
    dt_params = {
        'reg__estimator__max_depth': trial.suggest_int('reg__estimator__max_depth', 3, 15),
        'reg__estimator__min_samples_split': trial.suggest_int('reg__estimator__min_samples_split', 2, 20),
        'reg__estimator__min_samples_leaf': trial.suggest_int('reg__estimator__min_samples_leaf', 1, 10),
        'reg__estimator__max_features': trial.suggest_categorical('reg__estimator__max_features', ['sqrt', 'log2', None]),
    }

    base_dt = DecisionTreeRegressor(
        random_state=42
    )

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('reg', MultiOutputRegressor(base_dt))
    ])

    pipeline.set_params(**dt_params)
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    return r2_val

study_dt = optuna.create_study(direction='maximize')
study_dt.optimize(lambda trial: objective_dt(trial, X_train, y_train, X_val, y_val),
                  n_trials=50, show_progress_bar=True)

print("\n=== Decision Tree Best Parameters ===")
print(study_dt.best_params)
print(f"Best Validation Score: {study_dt.best_value:.4f}")

# --------------------------
# 5) Neural Network Optimization
# --------------------------
print("\n" + "=" * 60)
print("OPTIMIZING NEURAL NETWORK MODEL")
print("=" * 60)

def objective_ann(trial, X_train, y_train, X_val, y_val):
    # Define hidden layer architecture
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layers = []
    for i in range(n_layers):
        hidden_layers.append(trial.suggest_int(f'n_units_l{i}', 50, 200))
    
    # Define other hyperparameters
    alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    max_iter = trial.suggest_int('max_iter', 500, 1000)
    
    # Create model with suggested parameters directly
    base_ann = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        activation='relu',
        solver='adam',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('reg', MultiOutputRegressor(base_ann))
    ])

    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    return r2_val

study_ann = optuna.create_study(direction='maximize')
study_ann.optimize(lambda trial: objective_ann(trial, X_train, y_train, X_val, y_val),
                   n_trials=45, show_progress_bar=True)

print("\n=== Neural Network Best Parameters ===")
print(study_ann.best_params)
print(f"Best Validation Score: {study_ann.best_value:.4f}")

# --------------------------
# 6) Train all models with best parameters
# --------------------------
print("\n" + "=" * 60)
print("TRAINING ALL MODELS WITH BEST PARAMETERS")
print("=" * 60)

# XGBoost
final_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    booster='gbtree',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
pipeline_xgb = Pipeline([
    ('scaler', RobustScaler()),
    ('reg', MultiOutputRegressor(final_xgb))
])
pipeline_xgb.set_params(**study_xgb.best_params)
pipeline_xgb.fit(X_train, y_train)

# Random Forest
final_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
pipeline_rf = Pipeline([
    ('scaler', RobustScaler()),
    ('reg', MultiOutputRegressor(final_rf))
])
pipeline_rf.set_params(**study_rf.best_params)
pipeline_rf.fit(X_train, y_train)

# Decision Tree
final_dt = DecisionTreeRegressor(random_state=42)
pipeline_dt = Pipeline([
    ('scaler', RobustScaler()),
    ('reg', MultiOutputRegressor(final_dt))
])
pipeline_dt.set_params(**study_dt.best_params)
pipeline_dt.fit(X_train, y_train)

# Neural Network
# Extract best parameters from study
best_params_ann = study_ann.best_params
n_layers = best_params_ann['n_layers']
hidden_layers = tuple([best_params_ann[f'n_units_l{i}'] for i in range(n_layers)])

final_ann = MLPRegressor(
    hidden_layer_sizes=hidden_layers,
    alpha=best_params_ann['alpha'],
    learning_rate_init=best_params_ann['learning_rate_init'],
    max_iter=best_params_ann['max_iter'],
    activation='relu',
    solver='adam',
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
pipeline_ann = Pipeline([
    ('scaler', RobustScaler()),
    ('reg', MultiOutputRegressor(final_ann))
])
pipeline_ann.fit(X_train, y_train)

# --------------------------
# 7) Evaluate all models
# --------------------------
print("\n" + "=" * 60)
print("MODEL EVALUATION ON TEST SET")
print("=" * 60)

models = {
    'XGBoost': pipeline_xgb,
    'Random Forest': pipeline_rf,
    'Decision Tree': pipeline_dt,
    'Neural Network': pipeline_ann
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2_per_target = r2_score(y_test, y_pred, multioutput='raw_values')
    mae_per_target = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse_per_target = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    
    results[name] = {
        'predictions': y_pred,
        'r2_overall': r2,
        'mae_overall': mae,
        'mse_overall': mse,
        'r2_per_target': r2_per_target,
        'mae_per_target': mae_per_target,
        'mse_per_target': mse_per_target
    }
    
    print(f"\n{name} Results:")
    print(f"  Overall R²: {r2:.4f}")
    print(f"  Overall MAE: {mae:.4f}")
    print(f"  Overall MSE: {mse:.4f}")
    print(f"  Mean R² per target: {r2_per_target.mean():.4f}")
    print(f"  Mean MAE per target: {mae_per_target.mean():.4f}")
    print(f"  Mean MSE per target: {mse_per_target.mean():.4f}")

# --------------------------
# 8) Find the best model
# --------------------------
best_model_name = max(results.keys(), key=lambda k: results[k]['r2_overall'])
best_model = models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_model_name}")
print(f"R²: {results[best_model_name]['r2_overall']:.4f}")
print(f"MAE: {results[best_model_name]['mae_overall']:.4f}")
print(f"MSE: {results[best_model_name]['mse_overall']:.4f}")
print("=" * 60)

# --------------------------
# 9) Model Comparison Plots
# --------------------------
print("\n=== Creating Model Comparison Plots ===")

# Plot 1: Overall R² and MAE Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

model_names = list(results.keys())
r2_scores = [results[m]['r2_overall'] for m in model_names]
mae_scores = [results[m]['mae_overall'] for m in model_names]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars1 = ax1.bar(model_names, r2_scores, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{r2_scores[i]:.4f}', ha='center', va='bottom', fontweight='bold')
    if model_names[i] == best_model_name:
        bar.set_edgecolor('gold')
        bar.set_linewidth(3)

bars2 = ax2.bar(model_names, mae_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('Model Comparison - MAE', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
             f'{mae_scores[i]:.4f}', ha='center', va='bottom', fontweight='bold')
    if model_names[i] == best_model_name:
        bar.set_edgecolor('gold')
        bar.set_linewidth(3)

plt.tight_layout()
plt.show()

# Plot 2: R² per Target Comparison
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(y_target.columns))
width = 0.2

for i, (name, color) in enumerate(zip(model_names, colors)):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, results[name]['r2_per_target'], width, 
                  label=name, color=color, alpha=0.7, edgecolor='black')
    
    if name == best_model_name:
        for bar in bars:
            bar.set_edgecolor('gold')
            bar.set_linewidth(2)

ax.set_xlabel('Target Variables', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('R² Score Comparison Across Targets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(y_target.columns, rotation=45, ha='right')
ax.legend(loc='best')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 3: MAE per Target Comparison
fig, ax = plt.subplots(figsize=(16, 8))

for i, (name, color) in enumerate(zip(model_names, colors)):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, results[name]['mae_per_target'], width, 
                  label=name, color=color, alpha=0.7, edgecolor='black')
    
    if name == best_model_name:
        for bar in bars:
            bar.set_edgecolor('gold')
            bar.set_linewidth(2)

# تنظیمات فونت برای محورها و عنوان
ax.set_xlabel('Target Variables', fontsize=16, fontweight='bold')  # بزرگتر و بولد
ax.set_ylabel('MAE', fontsize=16, fontweight='bold')  # بزرگتر و بولد
ax.set_title('MAE Comparison Across Targets', fontsize=18, fontweight='bold', pad=20)  # بزرگتر

# تنظیم سایز فونت برای تیک‌های محورها
ax.set_xticks(x)
ax.set_xticklabels(y_target.columns, rotation=45, ha='right', fontsize=14)
ax.set_yticklabels(ax.get_yticks(), fontsize=14)

# تنظیم سایز فونت برای legend
ax.legend(loc='best', fontsize=13, title_fontsize=14)

# تنظیم سایز فونت برای اعداد روی محور y
ax.tick_params(axis='both', which='major', labelsize=14)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 4: Heatmap of R² scores
fig, ax = plt.subplots(figsize=(16, 8))

r2_matrix = np.array([results[m]['r2_per_target'] for m in model_names])

# تنظیم سایز فونت برای heatmap
sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            xticklabels=y_target.columns, yticklabels=model_names,
            cbar_kws={'label': 'R² Score', 'shrink': 0.8},
            annot_kws={'size': 13, 'weight': 'bold'},  # تنظیم سایز و وزن فونت annotها
            ax=ax, vmin=0, vmax=1)

# تنظیم سایز فونت برای tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

# تنظیم سایز فونت برای عنوان
ax.set_title('R² Score Heatmap - All Models & Targets', fontsize=16, fontweight='bold', pad=20)

# تنظیم سایز فونت برای label رنگ‌بار (colorbar)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_ylabel('R² Score', fontsize=12)

plt.tight_layout()
plt.show()
# --------------------------
# 10) Detailed Analysis for Best Model
# --------------------------
print(f"\n{'=' * 60}")
print(f"DETAILED ANALYSIS FOR BEST MODEL: {best_model_name}")
print(f"{'=' * 60}")

y_pred_best = best_predictions
r2_per_target = results[best_model_name]['r2_per_target']
mae_per_target = results[best_model_name]['mae_per_target']
mse_per_target = results[best_model_name]['mse_per_target']

# Metrics per target
print("\n=== Performance Metrics by Target ===")
for i, target in enumerate(y_target.columns):
    print(f"\n{target}:")
    print(f"  R²: {r2_per_target[i]:.4f}")
    print(f"  MAE: {mae_per_target[i]:.4f}")
    print(f"  MSE: {mse_per_target[i]:.4f}")

# Feature Importance (for tree-based models)
if best_model_name in ['XGBoost', 'Random Forest', 'Decision Tree']:
    print("\n=== Feature Importance Analysis ===")
    
    individual_models = best_model.named_steps['reg'].estimators_
    feature_importance_df = pd.DataFrame(index=X.columns)
    
    for i, (target, model) in enumerate(zip(y_target.columns, individual_models)):
        importance = model.feature_importances_
        feature_importance_df[target] = importance
    
    # Plot feature importance
    n_targets = len(y_target.columns)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    for i, target in enumerate(y_target.columns):
        top_10 = feature_importance_df[target].sort_values(ascending=False).head(10)
        axes[i].barh(range(len(top_10)), top_10.values, color='steelblue', alpha=0.7)
        axes[i].set_yticks(range(len(top_10)))
        axes[i].set_yticklabels(top_10.index)
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'Top 10 Features - {target}')
        axes[i].invert_yaxis()
        axes[i].grid(axis='x', alpha=0.3)
    
    for i in range(len(y_target.columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Predicted vs Actual Plots for Best Model
print("\n=== Creating Predicted vs Actual Plots ===")

n_cols = 3
n_rows = (len(y_target.columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))  # شکل بزرگتر

if len(y_target.columns) == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for i, target in enumerate(y_target.columns):
    y_true = y_test.iloc[:, i].values
    y_pred = y_pred_best[:, i]
    
    r2 = r2_per_target[i]
    mae = mae_per_target[i]
    
    axes[i].scatter(y_true, y_pred, alpha=0.7, s=80, color='steelblue', edgecolor='black', linewidth=0.5)  # نقاط بزرگتر
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')  # خط ضخیم‌تر
    
    # متن داخل باکس با فونت بزرگتر
    axes[i].text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}', 
                transform=axes[i].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5),
                fontsize=14, fontweight='bold')  # fontsize از 10 به 14 تغییر کرد
    
    axes[i].set_xlabel(f'Actual {target}', fontsize=14, fontweight='bold')  # از 10 به 14
    axes[i].set_ylabel(f'Predicted {target}', fontsize=14, fontweight='bold')  # از 10 به 14
    axes[i].set_title(f'{target}', fontsize=16, fontweight='bold', pad=12)  # از 11 به 16
    axes[i].legend(fontsize=12, loc='lower right')  # legend با فونت بزرگتر
    axes[i].grid(True, alpha=0.3, linestyle='--')
    
    # تنظیم سایز فونت برای تیک‌های محورها
    axes[i].tick_params(axis='both', which='major', labelsize=12)

for i in range(len(y_target.columns), len(axes)):
    axes[i].set_visible(False)

plt.suptitle(f'Predicted vs Actual - {best_model_name}', fontsize=20, fontweight='bold', y=1.02)  # از 16 به 20
plt.tight_layout()
plt.show()

# Performance Summary
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# R² by target
x = np.arange(len(y_target.columns))
bars = ax1.bar(x, r2_per_target, color='skyblue', edgecolor='navy', alpha=0.7)
ax1.set_xlabel('Targets')
ax1.set_ylabel('R² Score')
ax1.set_title(f'R² Score by Target - {best_model_name}')
ax1.set_xticks(x)
ax1.set_xticklabels(y_target.columns, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{r2_per_target[i]:.3f}', ha='center', va='bottom')

# MAE by target
bars2 = ax2.bar(x, mae_per_target, color='lightcoral', edgecolor='darkred', alpha=0.7)
ax2.set_xlabel('Targets')
ax2.set_ylabel('MAE')
ax2.set_title(f'MAE by Target - {best_model_name}')
ax2.set_xticks(x)
ax2.set_xticklabels(y_target.columns, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{mae_per_target[i]:.3f}', ha='center', va='bottom')

# Combined plot
ax3_twin = ax3.twinx()
bars_r2 = ax3.bar(x - 0.2, r2_per_target, 0.4, label='R²', color='blue', alpha=0.7)
bars_mae = ax3_twin.bar(x + 0.2, mae_per_target, 0.4, label='MAE', color='orange', alpha=0.7)

ax3.set_xlabel('Targets')
ax3.set_ylabel('R²', color='blue')
ax3_twin.set_ylabel('MAE', color='orange')
ax3.set_title(f'R² and MAE Comparison - {best_model_name}')
ax3.set_xticks(x)
ax3.set_xticklabels(y_target.columns, rotation=45, ha='right')
ax3.tick_params(axis='y', labelcolor='blue')
ax3_twin.tick_params(axis='y', labelcolor='orange')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Residuals plot
y_test_flat = y_test.values.flatten()
y_pred_flat = y_pred_best.flatten()
residuals = y_test_flat - y_pred_flat

ax4.scatter(y_pred_flat, residuals, alpha=0.6, s=20, color='purple')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.set_title(f'Residuals Plot - {best_model_name}')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------
# 11) Final Summary
# --------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nModel Rankings by R²:")
sorted_models = sorted(results.items(), key=lambda x: x[1]['r2_overall'], reverse=True)
for rank, (name, res) in enumerate(sorted_models, 1):
    print(f"{rank}. {name}: R² = {res['r2_overall']:.4f}, MAE = {res['mae_overall']:.4f}, MSE = {res['mse_overall']:.4f}")

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Overall R²: {results[best_model_name]['r2_overall']:.4f}")
print(f"   Overall MAE: {results[best_model_name]['mae_overall']:.4f}")
print(f"   Overall MSE: {results[best_model_name]['mse_overall']:.4f}")
print(f"   Best Target: {y_target.columns[np.argmax(r2_per_target)]} (R² = {r2_per_target.max():.4f})")
print(f"   Most Challenging Target: {y_target.columns[np.argmin(r2_per_target)]} (R² = {r2_per_target.min():.4f})")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)