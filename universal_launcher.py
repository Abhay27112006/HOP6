import os
import sys
import platform
import urllib.request
import zipfile
import shutil

# --- Configuration ---
# Replace this base URL with your actual hosting URL (e.g., GitHub Releases)
BASE_URL = "https://github.com/YOUR_USERNAME/6hops/releases/download/v1.0"

def get_platform_info():
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return "windows", "hop6_windows.zip"
    elif system == "darwin":
        return "mac", "hop6_mac.zip"
    elif system == "linux":
        return "linux", "hop6_linux.zip"
    else:
        print(f"Unsupported OS: {system}")
        sys.exit(1)

def download_file(url, dest):
    print(f"Downloading from {url} ...")
    try:
        # We add a User-Agent header in case the host requires it
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading file: {e}")
        print("\nPlease make sure the BASE_URL in this script points to your actual releases.")
        sys.exit(1)

def main():
    print("==================================================")
    print("       Hop6 Universal Downloader & Setup          ")
    print("==================================================")
    
    os_name, zip_name = get_platform_info()
    print(f"Detected OS: {os_name.capitalize()}")
    
    url = f"{BASE_URL}/{zip_name}"
    
    # Download to a temporary zip
    temp_zip = "hop6_temp_download.zip"
    download_file(url, temp_zip)
    
    print("Extracting files...")
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting ZIP: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
            
    print("\nSetup is ready! You can now launch Hop6:")
    if os_name == "windows":
        print(" -> Run: run.bat")
    else:
        print(" -> Run: ./run.sh")
    print("==================================================")

if __name__ == "__main__":
    main()
