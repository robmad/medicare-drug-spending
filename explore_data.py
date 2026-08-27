import zipfile
from pathlib import Path

# Your zip file
zip_path = Path("data") / "Monthly Prescription Drug Plan Formulary and Pharmacy Network Information.zip"

print(f"📦 Zip file: {zip_path.name}")
print(f"📊 File size: {zip_path.stat().st_size / (1024**3):.2f} GB")
print()

print("📂 Files inside the zip:")
print("=" * 80)

with zipfile.ZipFile(zip_path, 'r') as z:
    for info in z.filelist:
        size_mb = info.file_size / (1024**2)
        print(f"  {info.filename:<45} {size_mb:>10.1f} MB")
    
    print()
    print(f"✓ Total files: {len(z.filelist)}")
    total_uncompressed = sum(f.file_size for f in z.filelist) / (1024**3)
    print(f"✓ Total uncompressed size: {total_uncompressed:.2f} GB")
