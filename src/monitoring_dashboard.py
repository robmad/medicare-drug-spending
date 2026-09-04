"""
Prediction Monitoring Dashboard
Track model performance and predictions by month
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class MonitoringDashboard:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.monthly_dir = self.project_root / "data" / "raw" / "monthly_extracted"
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load the trained time-series model."""
        model_path = self.models_dir / "xgboost_timeseries.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded: {model_path}")
        return model
    
    def load_all_months(self):
        """Load all 13 months of data."""
        print("\nLoading all 13 months...")
        
        dfs = []
        
        for month_dir in sorted(self.monthly_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            
            month_key = month_dir.name
            
            formulary_files = [f for f in month_dir.glob("*basic drugs formulary*.txt") if 'sample' not in f.name.lower()]
            
            if not formulary_files:
                continue
            
            print(f"  {month_key}...", end=" ")
            
            bf_file = formulary_files[0]
            
            try:
                df = pd.read_csv(bf_file, sep='|', low_memory=False)
                
                year, month = month_key.split('-')
                df['YEAR_MONTH'] = month_key
                df['MONTH'] = int(month)
                df['YEAR'] = int(year)
                
                # Convert Y/N to 1/0
                for col in ['PRIOR_AUTHORIZATION_YN', 'STEP_THERAPY_YN', 'QUANTITY_LIMIT_YN', 'SELECTED_DRUG_YN']:
                    if col in df.columns:
                        df[col] = (df[col] == 'Y').astype(int)
                
                dfs.append(df)
                print("OK")
                
            except Exception as e:
                print(f"Error: {e}")
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Loaded {combined.shape[0]:,} rows")
        
        return combined
    
    def create_features(self, df):
        """Create the same features used in training."""
        df['QUANTITY_LIMIT_AMOUNT'] = pd.to_numeric(df['QUANTITY_LIMIT_AMOUNT'], errors='coerce')
        df['HAS_QTY_LIMIT_AMOUNT'] = (df['QUANTITY_LIMIT_AMOUNT'] > 0).astype(int)
        df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
        
        feature_cols = [
            'TIER_LEVEL_VALUE',
            'STEP_THERAPY_YN',
            'QUANTITY_LIMIT_YN',
            'SELECTED_DRUG_YN',
            'HAS_QTY_LIMIT_AMOUNT',
            'MONTH_SIN',
            'MONTH_COS'
        ]
        
        return df, feature_cols
    
    def make_predictions_by_month(self, df, model, feature_cols):
        """Make predictions for each month."""
        print("\nMaking predictions by month...")
        
        monthly_results = []
        
        for month in sorted(df['YEAR_MONTH'].unique()):
            month_data = df[df['YEAR_MONTH'] == month].copy()
            
            X = month_data[feature_cols].fillna(month_data[feature_cols].median())
            y = month_data['PRIOR_AUTHORIZATION_YN']
            
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            accuracy = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)
            prior_auth_actual = y.mean()
            prior_auth_pred = y_pred.mean()
            
            monthly_results.append({
                'Month': month,
                'Accuracy': accuracy,
                'ROC_AUC': auc,
                'Prior_Auth_Actual': prior_auth_actual,
                'Prior_Auth_Predicted': prior_auth_pred,
                'Samples': len(X),
                'Avg_Confidence': y_pred_proba.mean()
            })
            
            print(f"  {month}: Accuracy={accuracy:.4f}, AUC={auc:.4f}, Actual Prior Auth={prior_auth_actual*100:.2f}%")
        
        results_df = pd.DataFrame(monthly_results)
        print()
        
        return results_df
    
    def create_visualizations(self, results_df):
        """Create monitoring dashboard visualizations."""
        print("Creating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Monitoring Dashboard - 13 Months', fontsize=16, fontweight='bold')
        
        # 1. Accuracy over time
        ax1 = axes[0, 0]
        ax1.plot(results_df['Month'], results_df['Accuracy'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.axhline(y=results_df['Accuracy'].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy by Month')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC-AUC over time
        ax2 = axes[0, 1]
        ax2.plot(results_df['Month'], results_df['ROC_AUC'], marker='s', linewidth=2, markersize=8, color='#A23B72')
        ax2.axhline(y=results_df['ROC_AUC'].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_title('ROC-AUC Score by Month')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Prior Authorization Rate (Actual vs Predicted)
        ax3 = axes[1, 0]
        x = np.arange(len(results_df))
        width = 0.35
        ax3.bar(x - width/2, results_df['Prior_Auth_Actual']*100, width, label='Actual', color='#F18F01', alpha=0.8)
        ax3.bar(x + width/2, results_df['Prior_Auth_Predicted']*100, width, label='Predicted', color='#C73E1D', alpha=0.8)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Prior Authorization Rate (%)')
        ax3.set_title('Prior Auth Rate: Actual vs Predicted')
        ax3.set_xticks(x)
        ax3.set_xticklabels(results_df['Month'], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Model Confidence over time
        ax4 = axes[1, 1]
        ax4.plot(results_df['Month'], results_df['Avg_Confidence']*100, marker='^', linewidth=2, markersize=8, color='#06A77D')
        ax4.axhline(y=results_df['Avg_Confidence'].mean()*100, color='red', linestyle='--', alpha=0.7, label='Mean')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Prediction Confidence (%)')
        ax4.set_title('Model Confidence by Month')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim([0, 100])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = self.reports_dir / "monitoring_dashboard.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        
        plt.close()
    
    def generate_report(self, results_df):
        """Generate a text monitoring report."""
        print("Generating monitoring report...")
        
        report = []
        report.append("="*80)
        report.append("MODEL MONITORING REPORT - 13 MONTHS")
        report.append("="*80)
        report.append("")
        
        report.append("OVERALL METRICS:")
        report.append(f"  Average Accuracy: {results_df['Accuracy'].mean():.4f}")
        report.append(f"  Accuracy Std Dev: {results_df['Accuracy'].std():.4f}")
        report.append(f"  Accuracy Range: {results_df['Accuracy'].min():.4f} - {results_df['Accuracy'].max():.4f}")
        report.append("")
        
        report.append(f"  Average ROC-AUC: {results_df['ROC_AUC'].mean():.4f}")
        report.append(f"  ROC-AUC Std Dev: {results_df['ROC_AUC'].std():.4f}")
        report.append(f"  ROC-AUC Range: {results_df['ROC_AUC'].min():.4f} - {results_df['ROC_AUC'].max():.4f}")
        report.append("")
        
        report.append("DRIFT DETECTION:")
        accuracy_drift = results_df['Accuracy'].max() - results_df['Accuracy'].min()
        auc_drift = results_df['ROC_AUC'].max() - results_df['ROC_AUC'].min()
        
        report.append(f"  Accuracy Drift: {accuracy_drift:.4f} (LOW is good)")
        report.append(f"  ROC-AUC Drift: {auc_drift:.4f} (LOW is good)")
        
        if accuracy_drift < 0.01 and auc_drift < 0.01:
            report.append("  Status: STABLE - No significant drift detected")
        elif accuracy_drift < 0.05 or auc_drift < 0.05:
            report.append("  Status: ACCEPTABLE - Minor drift, monitor closely")
        else:
            report.append("  Status: WARNING - Significant drift detected!")
        
        report.append("")
        report.append("MONTHLY BREAKDOWN:")
        report.append("")
        
        for idx, row in results_df.iterrows():
            report.append(f"{row['Month']}:")
            report.append(f"  Accuracy: {row['Accuracy']:.4f}")
            report.append(f"  ROC-AUC: {row['ROC_AUC']:.4f}")
            report.append(f"  Prior Auth Rate (Actual): {row['Prior_Auth_Actual']*100:.2f}%")
            report.append(f"  Prior Auth Rate (Predicted): {row['Prior_Auth_Predicted']*100:.2f}%")
            report.append(f"  Samples: {row['Samples']:,}")
            report.append(f"  Avg Confidence: {row['Avg_Confidence']*100:.2f}%")
            report.append("")
        
        report.append("="*80)
        report.append("RECOMMENDATIONS:")
        report.append("="*80)
        report.append("1. Monitor accuracy and AUC weekly")
        report.append("2. Retrain model if drift exceeds 5%")
        report.append("3. Investigate if prior auth rates change >2%")
        report.append("4. Set up automated alerts for performance drops")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save
        report_path = self.reports_dir / "monitoring_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Saved: {report_path}")
        
        # Print to console
        print("\n" + report_text)
    
    def save_results_csv(self, results_df):
        """Save monthly results as CSV."""
        csv_path = self.reports_dir / "monthly_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved: {csv_path}")

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    dashboard = MonitoringDashboard(root)
    
    # Load model and data
    model = dashboard.load_model()
    df = dashboard.load_all_months()
    
    # Create features
    df, feature_cols = dashboard.create_features(df)
    
    # Make predictions by month
    results = dashboard.make_predictions_by_month(df, model, feature_cols)
    
    # Create visualizations
    dashboard.create_visualizations(results)
    
    # Generate report
    dashboard.generate_report(results)
    
    # Save results
    dashboard.save_results_csv(results)
    
    print("\n" + "="*80)
    print("SUCCESS! Monitoring dashboard complete.")
    print("="*80)
    print("\nOutputs saved to: reports/")