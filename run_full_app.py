import subprocess
import sys
import os
import time
import threading

def install_requirements():
    """Install required packages for both frontend and backend"""
    try:
        print("📦 Installing frontend requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Frontend requirements installed successfully!")
        
        print("📦 Installing backend requirements...")
        backend_req_path = os.path.join("scripts", "requirements.txt")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", backend_req_path])
        print("✅ Backend requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_backend():
    """Run the FastAPI backend server"""
    try:
        backend_path = os.path.join(os.getcwd(), "scripts", "main.py")
        print(f"🚀 Starting backend server from {backend_path}")
        subprocess.run([sys.executable, backend_path])
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error running backend: {e}")

def run_streamlit():
    """Run the Streamlit frontend application"""
    try:
        print("🚀 Starting Streamlit frontend...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error running frontend: {e}")

if __name__ == "__main__":
    print("🧠 Neural BCI Full Application Setup")
    print("=" * 50)
    
    # Install requirements
    if install_requirements():
        print("\n🚀 Starting Neural BCI Application...")
        
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend)
        backend_thread.daemon = True  # This ensures the thread will exit when the main program exits
        backend_thread.start()
        
        # Give the backend time to start up
        print("⏳ Waiting for backend to initialize...")
        time.sleep(5)
        
        print("\n📡 Backend API running at: http://localhost:8000")
        print("📡 API Documentation: http://localhost:8000/docs")
        print("📡 Frontend will open in your default browser at: http://localhost:8501")
        print("\n💡 Press Ctrl+C to stop both servers")
        print("=" * 50)
        
        # Run the frontend application (this will block until it exits)
        run_streamlit()
    else:
        print("❌ Failed to install requirements. Please check your Python environment.")