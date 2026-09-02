"""
ML Model Training - FIXED VERSION
Predicting Prior Authorization Requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import json

class DrugFormularyModel:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.project_root / "models"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_extracted_data(self, sample=True):
        """Load the extracted data files from pipeline."""
        extract_path = self.raw_dir / "extracted"
        
        print("\nLoading extracted data...")
        
        bf_files = list(extract_path.rglob("*basic drugs formulary*20260831.txt"))
        bc_files = list(extract_path.rglob("*beneficiary cost*20260831.txt"))
        
        bf_file = [f for f in bf_files if 'sample' not in f.name][0] if bf_files else None
        bc_file = [f for f in bc_files if 'sample' not in f.name][0] if bc_files else None
        
        if not bf_file or not bc_file:
            raise FileNotFoundError("Could not find formulary or cost files")
        
        print(f"Loading basic formulary: {bf_file.name}")
        nrows = 500000 if sample else None
        formulary = pd.read_csv(bf_file, sep='|', nrows=nrows, low_memory=False)
        
        print(f"Loading beneficiary cost: {bc_file.name}")
        cost = pd.read_csv(bc_file, sep='|', low_memory=False)
        
        print(f"Formulary shape: {formulary.shape}")
        print(f"Cost shape: {cost.shape}")
        
        print("\nConverting data types...")
        
        # Convert Y/N columns to 1/0
        yn_cols = ['PRIOR_AUTHORIZATION_YN', 'STEP_THERAPY_YN', 'QUANTITY_LIMIT_YN', 'SELECTED_DRUG_YN']
        for col in yn_cols:
            if col in formulary.columns:
                formulary[col] = (formulary[col] == 'Y').astype(int)
                print(f"  Converted {col}: Y/N -> 1/0")
        
        # Convert numeric columns
        numeric_cols = ['FORMULARY_ID', 'FORMULARY_VERSION', 'CONTRACT_YEAR', 'RXCUI', 'NDC', 
                       'TIER_LEVEL_VALUE', 'QUANTITY_LIMIT_AMOUNT', 'QUANTITY_LIMIT_DAYS']
        
        for col in numeric_cols:
            if col in formulary.columns:
                formulary[col] = pd.to_numeric(formulary[col], errors='coerce')
        
        print("  Conversion complete!")
        
        return formulary, cost
    
    def create_features(self, formulary, cost):
        """Create ML-ready features."""
        print("\n" + "="*80)
        print("CREATING FEATURES")
        print("="*80)
        
        df = formulary.copy()
        
        print(f"\nTarget variable: PRIOR_AUTHORIZATION_YN")
        no_prior = (df['PRIOR_AUTHORIZATION_YN'] == 0).sum()
        yes_prior = (df['PRIOR_AUTHORIZATION_YN'] == 1).sum()
        total = no_prior + yes_prior
        
        print(f"  Class distribution:")
        print(f"    0 (No prior auth): {no_prior:,} ({no_prior/total:.1%})")
        print(f"    1 (Requires prior auth): {yes_prior:,} ({yes_prior/total:.1%})")
        
        print(f"\nFeature engineering:")
        df['HAS_QTY_LIMIT_AMOUNT'] = (df['QUANTITY_LIMIT_AMOUNT'] > 0).astype(int)
        print(f"  + HAS_QTY_LIMIT_AMOUNT (derived)")
        
        feature_cols = [
            'TIER_LEVEL_VALUE',
            'STEP_THERAPY_YN',
            'QUANTITY_LIMIT_YN',
            'HAS_QTY_LIMIT_AMOUNT',
            'SELECTED_DRUG_YN',
            'CONTRACT_YEAR'
        ]
        
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        print(f"\nFinal features ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"  - {col}")
        
        return df, feature_cols
    
    def train_model(self, df, feature_cols, test_size=0.2, random_state=42):
        """Train XGBoost model."""
        print("\n" + "="*80)
        print("TRAINING XGBOOST MODEL")
        print("="*80)
        
        X = df[feature_cols].copy()
        y = df['PRIOR_AUTHORIZATION_YN'].copy()
        
        # Remove NaN rows
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"\nData shape: {X.shape}")
        print(f"Handling missing values...")
        print(f"  Missing before: {X.isnull().sum().sum()}")
        X = X.fillna(X.median(numeric_only=True))
        print(f"  Missing after: {X.isnull().sum().sum()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Train: {X_train.shape[0]:,} samples ({y_train.mean():.1%} prior auth)")
        print(f"  Test: {X_test.shape[0]:,} samples ({y_test.mean():.1%} prior auth)")
        
        print(f"\nTraining XGBoost...")
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            verbose=0
        )
        
        model.fit(X_train, y_train, verbose=False)
        print("Training complete!")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Prior Auth', 'Prior Auth']))
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        accuracy = (y_pred == y_test).mean()
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc_score:.4f}")
        
        print(f"\n" + "="*80)
        print("FEATURE IMPORTANCE")
        print("="*80 + "\n")
        
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance.iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")
        
        model_path = self.models_dir / "xgboost_prior_auth.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to: {model_path}")
        
        metrics = {
            'auc_score': float(auc_score),
            'accuracy': float(accuracy),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': feature_cols
        }
        
        metrics_path = self.models_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        return model, metrics

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    model_trainer = DrugFormularyModel(project_root)
    formulary, cost = model_trainer.load_extracted_data(sample=True)
    df_features, feature_cols = model_trainer.create_features(formulary, cost)
    model, metrics = model_trainer.train_model(df_features, feature_cols)
    print("\n" + "="*80)
    print("SUCCESS! Model trained and saved.")
    print("="*80)

