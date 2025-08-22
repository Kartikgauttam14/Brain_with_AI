import asyncio
import websockets
import json
import numpy as np
import time
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoSimulator:
    def __init__(self, server_url="ws://localhost:8000/ws/arduino"):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.session_id = 1
        self.sampling_rate = 250
        self.channels = 8
        self.buffer_size = 125  # 0.5 second window
        self.eeg_buffer = [[] for _ in range(self.channels)]
        
    async def connect(self):
        """Connect to the backend server"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info(f"Connected to server at {self.server_url}")
            
            # Send initial status
            await self.send_status()
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            self.is_connected = False
    
    async def disconnect(self):
        """Disconnect from server"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from server")
    
    async def send_status(self):
        """Send status update to server"""
        status_message = {
            "type": "status",
            "timestamp": int(time.time() * 1000),
            "battery_level": 85.0,
            "signal_quality": 0.8,
            "sampling_rate": self.sampling_rate,
            "channels": self.channels,
            "is_recording": True
        }
        
        if self.websocket and self.is_connected:
            await self.websocket.send(json.dumps(status_message))
    
    def generate_eeg_sample(self, class_type="relax"):
        """Generate a single EEG sample for all channels"""
        sample = []
        
        for ch in range(self.channels):
            if class_type == "blink":
                # Strong frontal activity for blink
                if ch < 2:  # Frontal channels
                    amplitude = np.random.normal(100e-6, 20e-6)
                else:
                    amplitude = np.random.normal(10e-6, 5e-6)
            elif class_type == "focus":
                # Increased beta activity
                beta_component = 25e-6 * np.sin(2 * np.pi * 20 * time.time())
                amplitude = beta_component + np.random.normal(0, 8e-6)
            elif class_type == "relax":
                # Strong alpha activity
                alpha_component = 40e-6 * np.sin(2 * np.pi * 10 * time.time())
                amplitude = alpha_component + np.random.normal(0, 5e-6)
            elif class_type == "left_motor":
                # Right hemisphere activity
                if ch >= self.channels // 2:
                    amplitude = np.random.normal(30e-6, 10e-6)
                else:
                    amplitude = np.random.normal(15e-6, 5e-6)
            elif class_type == "right_motor":
                # Left hemisphere activity
                if ch < self.channels // 2:
                    amplitude = np.random.normal(30e-6, 10e-6)
                else:
                    amplitude = np.random.normal(15e-6, 5e-6)
            else:
                # Default noise
                amplitude = np.random.normal(0, 10e-6)
            
            sample.append(float(amplitude))
        
        return sample
    
    async def send_eeg_data(self):
        """Send EEG data buffer to server"""
        if len(self.eeg_buffer[0]) >= self.buffer_size:
            eeg_message = {
                "type": "eeg_data",
                "session_id": self.session_id,
                "timestamp": int(time.time() * 1000),
                "data": [channel_data[-self.buffer_size:] for channel_data in self.eeg_buffer],
                "sampling_rate": self.sampling_rate,
                "signal_quality": np.random.uniform(0.7, 0.95)
            }
            
            if self.websocket and self.is_connected:
                await self.websocket.send(json.dumps(eeg_message))
                logger.info(f"Sent EEG data buffer ({self.buffer_size} samples per channel)")
            
            # Clear buffer after sending
            self.eeg_buffer = [[] for _ in range(self.channels)]
    
    async def listen_for_commands(self):
        """Listen for commands from server"""
        try:
            while self.is_connected:
                if self.websocket:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    try:
                        data = json.loads(message)
                        if data.get("type") == "command":
                            command = data.get("command", "NONE")
                            confidence = data.get("confidence", 0.0)
                            logger.info(f"Received command: {command} (confidence: {confidence:.2f})")
                            
                            # Simulate command execution
                            await self.execute_command(command)
                            
                    except json.JSONDecodeError:
                        logger.error("Received invalid JSON from server")
                        
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue loop
        except Exception as e:
            logger.error(f"Error listening for commands: {str(e)}")
    
    async def execute_command(self, command):
        """Simulate command execution"""
        execution_time = np.random.uniform(10, 50)  # ms
        success = np.random.random() > 0.1  # 90% success rate
        
        logger.info(f"Executing command: {command} ({'SUCCESS' if success else 'FAILED'})")
        
        # Send execution result back to server
        result_message = {
            "type": "command_result",
            "command": command,
            "success": success,
            "execution_time_ms": execution_time,
            "timestamp": int(time.time() * 1000)
        }
        
        if self.websocket and self.is_connected:
            await self.websocket.send(json.dumps(result_message))
    
    async def simulate_eeg_recording(self):
        """Main simulation loop"""
        class_sequence = ["relax", "focus", "blink", "left_motor", "right_motor"]
        class_index = 0
        samples_in_class = 0
        samples_per_class = 250  # 1 second per class at 250 Hz
        
        while self.is_connected:
            try:
                # Generate current class type
                current_class = class_sequence[class_index]
                
                # Generate EEG sample
                sample = self.generate_eeg_sample(current_class)
                
                # Add to buffer
                for ch in range(self.channels):
                    self.eeg_buffer[ch].append(sample[ch])
                
                samples_in_class += 1
                
                # Switch class after specified number of samples
                if samples_in_class >= samples_per_class:
                    class_index = (class_index + 1) % len(class_sequence)
                    samples_in_class = 0
                    logger.info(f"Switching to class: {class_sequence[class_index]}")
                
                # Send data when buffer is full
                if len(self.eeg_buffer[0]) >= self.buffer_size:
                    await self.send_eeg_data()
                
                # Send status update every 5 seconds
                if int(time.time()) % 5 == 0 and samples_in_class == 0:
                    await self.send_status()
                
                # Simulate sampling rate
                await asyncio.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {str(e)}")
                break
    
    async def run(self):
        """Run the Arduino simulator"""
        await self.connect()
        
        if self.is_connected:
            # Start concurrent tasks
            tasks = [
                asyncio.create_task(self.simulate_eeg_recording()),
                asyncio.create_task(self.listen_for_commands())
            ]
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.info("Simulation interrupted by user")
            except Exception as e:
                logger.error(f"Simulation error: {str(e)}")
            finally:
                # Cancel tasks
                for task in tasks:
                    task.cancel()
                
                await self.disconnect()

async def main():
    """Main function to run the simulator"""
    simulator = ArduinoSimulator()
    
    try:
        await simulator.run()
    except KeyboardInterrupt:
        logger.info("Arduino simulator stopped")

if __name__ == "__main__":
    asyncio.run(main())
