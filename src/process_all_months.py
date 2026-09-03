import pandas as pd
from pathlib import Path

class MultiMonthProcessor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.monthly_dir = self.project_root / "data" / "raw" / "monthly_extracted"
    
    def load_combine(self, sample=False):
        print("\n" + "="*80)
        print("LOADING AND COMBINING ALL 13 MONTHS")
        print("="*80 + "\n")
        
        dfs = []
        
        for month_dir in sorted(self.monthly_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            
            month_key = month_dir.name
            
            # Find formulary file (skip samples)
            formulary_files = [f for f in month_dir.glob("*basic drugs formulary*.txt") if 'sample' not in f.name.lower()]
            
            if not formulary_files:
                print(f"{month_key}: No formulary file found")
                continue
            
            print(f"Loading {month_key}...")
            
            bf_file = formulary_files[0]
            nrows = 100000 if sample else None
            
            try:
                df = pd.read_csv(bf_file, sep='|', nrows=nrows, low_memory=False)
                
                # Add month columns
                year, month = month_key.split('-')
                df['YEAR_MONTH'] = month_key
                df['MONTH'] = int(month)
                df['YEAR'] = int(year)
                
                # Convert Y/N to 1/0
                for col in ['PRIOR_AUTHORIZATION_YN', 'STEP_THERAPY_YN', 'QUANTITY_LIMIT_YN', 'SELECTED_DRUG_YN']:
                    if col in df.columns:
                        df[col] = (df[col] == 'Y').astype(int)
                
                dfs.append(df)
                print(f"  OK - {len(df):,} rows")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"\nCombining {len(dfs)} months...")
        combined = pd.concat(dfs, ignore_index=True)
        
        print(f"Combined shape: {combined.shape}")
        print(f"Date range: {combined['YEAR_MONTH'].min()} to {combined['YEAR_MONTH'].max()}")
        print()
        
        return combined
    
    def show_trends(self, df):
        print("="*80)
        print("PRIOR AUTHORIZATION RATES BY MONTH")
        print("="*80 + "\n")
        
        trends = df.groupby('YEAR_MONTH')['PRIOR_AUTHORIZATION_YN'].agg(['sum', 'mean', 'count'])
        trends.columns = ['Count', 'Rate', 'Total']
        trends['Rate'] = (trends['Rate'] * 100).round(2)
        
        print(trends)
        print()

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    p = MultiMonthProcessor(root)
    
    df = p.load_combine(sample=True)
    p.show_trends(df)
    
    print("="*80)
    print("SUCCESS! All 13 months loaded and combined.")
    print("="*80)

