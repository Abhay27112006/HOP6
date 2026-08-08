import zipfile
import os

def create_bundle():
    bundle_name = "hop6_universal_bundle.zip"
    files_to_include = [
        "universal_launcher.py"
    ]
    
    print(f"Creating {bundle_name}...")
    with zipfile.ZipFile(bundle_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file}")
            else:
                print(f"Warning: {file} not found.")
                
    print("Done! You can distribute hop6_universal_bundle.zip to your users.")

if __name__ == "__main__":
    create_bundle()
