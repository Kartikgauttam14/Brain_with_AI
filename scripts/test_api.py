import requests
import json
import time
import asyncio
import websockets
from typing import Dict, Any

class NeuralBCIAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.session_id = None
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("Testing health check...")
        response = requests.get(f"{self.base_url}/api/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    def test_user_registration(self):
        """Test user registration"""
        print("Testing user registration...")
        user_data = {
            "username": f"testuser_{int(time.time())}",
            "email": f"test_{int(time.time())}@example.com",
            "password": "testpassword123"
        }
        
        response = requests.post(f"{self.base_url}/api/auth/register", json=user_data)
        print(f"Registration: {response.status_code} - {response.json()}")
        
        if response.status_code == 200:
            self.test_username = user_data["username"]
            self.test_password = user_data["password"]
            return True
        return False
    
    def test_user_login(self):
        """Test user login"""
        print("Testing user login...")
        login_data = {
            "username": self.test_username,
            "password": self.test_password
        }
        
        response = requests.post(
            f"{self.base_url}/api/auth/login",
            data=login_data
        )
        print(f"Login: {response.status_code} - {response.json()}")
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            return True
        return False
    
    def get_headers(self):
        """Get authorization headers"""
        return {"Authorization": f"Bearer {self.token}"}
    
    def test_create_session(self):
        """Test creating EEG session"""
        print("Testing session creation...")
        session_data = {
            "session_type": "testing",
            "duration_seconds": 300,
            "notes": "API test session"
        }
        
        response = requests.post(
            f"{self.base_url}/api/sessions",
            json=session_data,
            headers=self.get_headers()
        )
        print(f"Session creation: {response.status_code} - {response.json()}")
        
        if response.status_code == 200:
            self.session_id = response.json()["session_id"]
            return True
        return False
    
    def test_upload_eeg_data(self):
        """Test uploading EEG data"""
        print("Testing EEG data upload...")
        
        # Generate sample EEG data
        import numpy as np
        eeg_batch = {
            "session_id": self.session_id,
            "data_points": []
        }
        
        for i in range(10):  # 10 data points
            data_point = {
                "timestamp_ms": int(time.time() * 1000) + i * 4,  # 250 Hz sampling
                "channel_data": np.random.normal(0, 10e-6, 8).tolist(),
                "signal_quality": 0.8 + np.random.random() * 0.2
            }
            eeg_batch["data_points"].append(data_point)
        
        response = requests.post(
            f"{self.base_url}/api/data/batch",
            json=eeg_batch,
            headers=self.get_headers()
        )
        print(f"EEG data upload: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    def test_make_prediction(self):
        """Test making prediction"""
        print("Testing prediction...")
        
        # Generate sample EEG window (8 channels x 125 samples)
        import numpy as np
        eeg_window = []
        for ch in range(8):
            # Generate 0.5 second of data (125 samples at 250 Hz)
            channel_data = np.random.normal(0, 20e-6, 125).tolist()
            eeg_window.append(channel_data)
        
        response = requests.post(
            f"{self.base_url}/api/predict",
            params={"session_id": self.session_id},
            json=eeg_window,
            headers=self.get_headers()
        )
        print(f"Prediction: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    def test_get_system_status(self):
        """Test system status"""
        print("Testing system status...")
        response = requests.get(
            f"{self.base_url}/api/status",
            headers=self.get_headers()
        )
        print(f"System status: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    def test_get_sessions(self):
        """Test getting user sessions"""
        print("Testing get sessions...")
        response = requests.get(
            f"{self.base_url}/api/sessions",
            headers=self.get_headers()
        )
        print(f"Get sessions: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    def test_session_analytics(self):
        """Test session analytics"""
        print("Testing session analytics...")
        response = requests.get(
            f"{self.base_url}/api/analytics/session/{self.session_id}",
            headers=self.get_headers()
        )
        print(f"Session analytics: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    def test_train_model(self):
        """Test model training"""
        print("Testing model training...")
        response = requests.post(
            f"{self.base_url}/api/train",
            headers=self.get_headers()
        )
        print(f"Model training: {response.status_code} - {response.json()}")
        return response.status_code == 200
    
    async def test_websocket_connection(self):
        """Test WebSocket connection"""
        print("Testing WebSocket connection...")
        try:
            uri = f"ws://localhost:8000/ws/realtime"
            async with websockets.connect(uri) as websocket:
                # Send test message
                test_message = {"type": "test", "message": "Hello WebSocket"}
                await websocket.send(json.dumps(test_message))
                
                # Receive response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"WebSocket response: {response}")
                return True
        except Exception as e:
            print(f"WebSocket test failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 50)
        print("Neural BCI API Test Suite")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("User Registration", self.test_user_registration),
            ("User Login", self.test_user_login),
            ("Create Session", self.test_create_session),
            ("Upload EEG Data", self.test_upload_eeg_data),
            ("Make Prediction", self.test_make_prediction),
            ("System Status", self.test_get_system_status),
            ("Get Sessions", self.test_get_sessions),
            ("Session Analytics", self.test_session_analytics),
            ("Train Model", self.test_train_model),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = "PASS" if result else "FAIL"
                print(f"{test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results[test_name] = f"ERROR: {str(e)}"
                print(f"{test_name}: ERROR - {str(e)}")
            print("-" * 30)
        
        # Test WebSocket
        try:
            ws_result = asyncio.run(self.test_websocket_connection())
            results["WebSocket"] = "PASS" if ws_result else "FAIL"
            print(f"WebSocket: {'PASS' if ws_result else 'FAIL'}")
        except Exception as e:
            results["WebSocket"] = f"ERROR: {str(e)}"
            print(f"WebSocket: ERROR - {str(e)}")
        
        print("=" * 50)
        print("Test Results Summary:")
        for test_name, result in results.items():
            print(f"{test_name}: {result}")
        
        passed = sum(1 for r in results.values() if r == "PASS")
        total = len(results)
        print(f"\nPassed: {passed}/{total} tests")
        print("=" * 50)

if __name__ == "__main__":
    tester = NeuralBCIAPITester()
    tester.run_all_tests()
