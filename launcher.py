import os
import sys
import platform
import subprocess
import time

def check_venv():
    return os.path.exists(".venv") or os.path.exists("venv")

def main():
    print("=============================================")
    print("        Hop6 Engine Launcher                 ")
    print("=============================================")
    
    if not check_venv():
        print("Error: Virtual environment not found.")
        print("Please run setup_env.bat (Windows) or setup_env.sh (Mac/Linux) first.")
        time.sleep(3)
        return

    sys_platform = platform.system()
    
    if sys_platform == "Windows":
        # Check both possible venv names
        if os.path.exists(".venv"):
            activate_cmd = r".\.venv\Scripts\activate.bat"
        else:
            activate_cmd = r".\venv\Scripts\activate.bat"
            
        cmd = f"call {activate_cmd} && python src\\cli.py"
        os.system(cmd)
    else:
        if os.path.exists(".venv"):
            activate_cmd = "source .venv/bin/activate"
        else:
            activate_cmd = "source venv/bin/activate"
            
        cmd = f"bash -c '{activate_cmd} && python src/cli.py'"
        os.system(cmd)

if __name__ == "__main__":
    main()
