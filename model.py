import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, make_scorer)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    df = pd.read_csv('german_credit_data.csv')
    
    # Handle missing values
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    
    # Create target variable with more balanced classes
    conditions = [
        (df['Credit amount'] <= 4000) & 
        (df['Duration'] <= 18) &
        (df['Saving accounts'].isin(['quite rich', 'rich'])),
        
        (df['Credit amount'] > 7000) |
        (df['Duration'] > 36) |
        (df['Saving accounts'].isin(['little']))
    ]
    df['Risk'] = np.select(conditions, [1, 0], default=1)  # 1=Good, 0=Bad
    
    return df

def train_optimized_model():
    df = load_data()
    
    # Feature engineering
    df['Debt_to_Income'] = df['Credit amount'] / (df['Duration'] + 1)
    df['Age_Group'] = pd.cut(df['Age'], bins=[18,25,35,45,60,100], 
                            labels=['18-25','26-35','36-45','46-60','60+'])
    
    # Define features
    numeric_features = ['Age', 'Job', 'Credit amount', 'Duration', 'Debt_to_Income']
    categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_Group']
    
    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Split data
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline with SMOTE and classifier
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    # Hyperparameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    # Custom scoring metric (focus on recall for bad risk detection)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'custom_score': make_scorer(recall_score, pos_label=0)  # Focus on catching bad risks
    }
    
    # Set up cross-validated grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit='custom_score',  # Optimize for catching bad risks
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Generate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'bad_risk_recall': recall_score(y_test, y_pred, pos_label=0)
    }
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['High Risk', 'Low Risk'],
                yticklabels=['High Risk', 'Low Risk'])
    plt.title('Optimized Model Confusion Matrix\n(0: High Risk, 1: Low Risk)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('optimized_confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_model.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title('Top 15 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Print evaluation results
    print("\n" + "="*50)
    print("Optimized Model Evaluation Results")
    print("="*50)
    print("\nBest Parameters:")
    print(grid_search.best_params_)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nKey Metrics:")
    for name, value in metrics.items():
        print(f"{name:>15}: {value:.4f}")
    
    # Save model
    joblib.dump({
        'model': best_model,
        'preprocessor': preprocessor,
        'metrics': metrics,
        'feature_importance': feat_imp,
        'best_params': grid_search.best_params_
    }, 'optimized_credit_model.pkl')
    print("\nOptimized model saved as optimized_credit_model.pkl")

if __name__ == "__main__":
    train_optimized_model()