#!/usr/bin/env python3
"""
Installation script for FinOps AI Agent dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required dependencies"""
    print("🚀 Installing FinOps AI Agent dependencies...")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt file not found!")
        return
    
    # Read requirements from file
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    print(f"📦 Found {len(requirements)} packages to install")
    print()
    
    failed_packages = []
    
    for requirement in requirements:
        print(f"📥 Installing {requirement}...")
        if not install_package(requirement):
            failed_packages.append(requirement)
        print()
    
    print("=" * 50)
    
    if failed_packages:
        print(f"❌ Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"   • {package}")
        print("\n💡 Please try installing them manually using:")
        print("   pip install <package_name>")
    else:
        print("🎉 All dependencies installed successfully!")
        print("\n🚀 You can now run the FinOps AI Agent:")
        print("   streamlit run enhanced_streamlit_app.py")

if __name__ == "__main__":
    main() 