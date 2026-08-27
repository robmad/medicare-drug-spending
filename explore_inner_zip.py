import zipfile
from pathlib import Path

# The outer zip file
outer_zip = Path("data") / "Monthly Prescription Drug Plan Formulary and Pharmacy Network Information.zip"

print("Extracting inner zip from outer zip...")
print()

# Extract the inner zip
with zipfile.ZipFile(outer_zip, 'r') as outer:
    # Find the inner zip
    inner_zip_name = None
    for file in outer.namelist():
        if file.endswith('.zip') and '2026_' in file:
            inner_zip_name = file
            break
    
    if inner_zip_name:
        print(f"✓ Found inner zip: {inner_zip_name}")
        
        # Extract it
        inner_zip_content = outer.read(inner_zip_name)
        inner_zip_path = Path("data") / "2026_20260722.zip"
        
        # Write it
        with open(inner_zip_path, 'wb') as f:
            f.write(inner_zip_content)
        
        print(f"✓ Extracted to: {inner_zip_path}")
        print(f"✓ Size: {inner_zip_path.stat().st_size / (1024**3):.2f} GB")
        print()
        
        # Now list contents of inner zip
        print("📂 Files inside the inner zip (THE REAL DATA):")
        print("=" * 80)
        
        with zipfile.ZipFile(inner_zip_path, 'r') as inner:
            for info in inner.filelist:
                size_mb = info.file_size / (1024**2)
                print(f"  {info.filename:<45} {size_mb:>10.1f} MB")
            
            print()
            print(f"✓ Total files: {len(inner.filelist)}")
            total_uncompressed = sum(f.file_size for f in inner.filelist) / (1024**3)
            print(f"✓ Total uncompressed size: {total_uncompressed:.2f} GB")

