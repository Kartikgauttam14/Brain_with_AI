import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit application"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    print("🧠 Neural BCI Dashboard Setup")
    print("=" * 40)
    
    # Install requirements
    print("📦 Installing requirements...")
    if install_requirements():
        print("\n🚀 Starting Neural BCI Dashboard...")
        print("📡 Dashboard will open in your default browser")
        print("🔗 URL: http://localhost:8501")
        print("\n💡 Press Ctrl+C to stop the server")
        print("=" * 40)
        
        # Run the application
        run_streamlit()
    else:
        print("❌ Failed to install requirements. Please check your Python environment.")
