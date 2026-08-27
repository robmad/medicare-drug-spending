"""
Data Pipeline for Medicare Drug Spending Project
"""

import zipfile
from pathlib import Path
import pandas as pd

class DataPipeline:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_all_zips(self, zip_file):
        """Recursively extract all nested zips."""
        extract_path = self.raw_dir / "extracted"
        extract_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Extracting {zip_file.name}...")
        
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_path)
        
        iteration = 0
        while True:
            iteration += 1
            nested_zips = list(extract_path.rglob('*.zip'))
            if not nested_zips:
                print(f"✓ All zips extracted (after {iteration} iterations)")
                break
            
            print(f"  Iteration {iteration}: Found {len(nested_zips)} zip files")
            for nested_zip in nested_zips:
                print(f"    Extracting {nested_zip.name}...")
                try:
                    with zipfile.ZipFile(nested_zip, 'r') as z:
                        z.extractall(nested_zip.parent)
                    nested_zip.unlink()
                except Exception as e:
                    print(f"    ⚠️  Error: {e}")
        
        return extract_path
    
    def find_data_files(self, extract_path):
        """Find all pipe-delimited data files."""
        print("\n📂 Finding data files...")
        
        txt_files = {}
        for txt_file in extract_path.rglob('*.txt'):
            size_mb = txt_file.stat().st_size / (1024**2)
            
            if 'sample' in txt_file.name.lower():
                continue
            
            print(f"  {txt_file.name:<50} {size_mb:>10.1f} MB")
            
            name = txt_file.name.lower()
            if 'plan' in name and 'formulary' not in name:
                txt_files['plan_info'] = txt_file
            elif 'basic' in name or 'formulary' in name:
                if 'excluded' not in name and 'indication' not in name:
                    txt_files['basic_formulary'] = txt_file
            elif 'beneficiary' in name and 'insulin' not in name:
                txt_files['beneficiary_cost'] = txt_file
            elif 'pharmacy' in name or 'network' in name:
                if 'pharmacy_network' not in txt_files:
                    txt_files['pharmacy_network'] = []
                txt_files['pharmacy_network'].append(txt_file)
            elif 'geo' in name:
                txt_files['geo_locator'] = txt_file
        
        print(f"✓ Found {len(txt_files)} file categories")
        return txt_files
    
    def load_pipe_delimited(self, filepath, nrows=None):
        """Load a pipe-delimited file."""
        try:
            df = pd.read_csv(filepath, sep='|', nrows=nrows)
            print(f"  ✓ {filepath.name}: {len(df):,} rows × {len(df.columns)} cols")
            return df
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            return None
    
    def process_month(self, zip_file, sample=True):
        """Full pipeline: extract, load, prepare."""
        print(f"\n{'='*80}")
        print(f"Processing {zip_file.name}")
        print(f"{'='*80}\n")
        
        extract_path = self.extract_all_zips(zip_file)
        data_files = self.find_data_files(extract_path)
        
        print("\n" + "="*80)
        print("LOADING DATA FILES")
        print("="*80 + "\n")
        
        nrows = 100000 if sample else None
        
        dfs = {}
        
        # Load each file type
        if 'plan_info' in data_files:
            print("Plan Information:")
            dfs['plan_info'] = self.load_pipe_delimited(data_files['plan_info'])
        
        if 'basic_formulary' in data_files:
            print("\nBasic Formulary (sampling 100k rows):")
            dfs['basic_formulary'] = self.load_pipe_delimited(data_files['basic_formulary'], nrows=nrows)
        
        if 'beneficiary_cost' in data_files:
            print("\nBeneficiary Cost:")
            dfs['beneficiary_cost'] = self.load_pipe_delimited(data_files['beneficiary_cost'])
        
        if 'pharmacy_network' in data_files:
            print(f"\nPharmacy Network (6 parts, loading first):")
            dfs['pharmacy_network'] = self.load_pipe_delimited(data_files['pharmacy_network'][0], nrows=nrows)
        
        if 'geo_locator' in data_files:
            print("\nGeographic Locator:")
            dfs['geo_locator'] = self.load_pipe_delimited(data_files['geo_locator'])
        
        # Show column names
        print("\n" + "="*80)
        print("BASIC FORMULARY COLUMNS (Target variable candidates)")
        print("="*80 + "\n")
        
        if 'basic_formulary' in dfs and dfs['basic_formulary'] is not None:
            bf = dfs['basic_formulary']
            for i, col in enumerate(bf.columns):
                col_type = 'Y/N' if bf[col].dtype == 'object' else 'numeric'
                print(f"  {col:<40} ({col_type})")
        
        print("\n✅ Pipeline complete!")
        return {'dataframes': dfs, 'extract_path': extract_path}

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    pipeline = DataPipeline(project_root)
    
    zip_file = project_root / "data" / "2026_20260722.zip"
    
    if zip_file.exists():
        result = pipeline.process_month(zip_file, sample=True)
    else:
        print(f"❌ Zip file not found: {zip_file}")

