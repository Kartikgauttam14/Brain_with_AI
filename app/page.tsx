\`\`\`python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import time
import json
from datetime import datetime, timedelta
import asyncio
import websockets
import threading
from typing import Dict, List, Optional
import logging

# Configure page
st.set_page_config(
    page_title="Neural BCI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f2937;
    font-size: 2.5rem;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #3b82f6, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.status-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border-left: 4px solid #3b82f6;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #1f2937;
}

.metric-label {
    font-size: 0.875rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.alert-success {
    background-color: #d1fae5;
    border: 1px solid #a7f3d0;
    border-radius: 0.375rem;
    padding: 0.75rem;
    color: #065f46;
}

.alert-danger {
    background-color: #fee2e2;
    border: 1px solid #fecaca;
    border-radius: 0.375rem;
    padding: 0.75rem;
    color: #991b1b;
}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/realtime"

# Initialize session state
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False
if 'signal_strength' not in st.session_state:
    st.session_state.signal_strength = 0.0
if 'current_command' not in st.session_state:
    st.session_state.current_command = 'None'
if 'battery_level' not in st.session_state:
    st.session_state.battery_level = 85.0
if 'eeg_data' not in st.session_state:
    st.session_state.eeg_data = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

class BCIDataManager:
    def __init__(self):
        self.api_base = API_BASE_URL
        self.token = None
        
    def get_system_status(self) -> Dict:
        """Get system status from API"""
        try:
            response = requests.get(f"{self.api_base}/api/status")
            if response.status_code == 200:
                return response.json()
            else:
                return self.get_mock_status()
        except requests.RequestException:
            return self.get_mock_status()
    
    def get_mock_status(self) -> Dict:
        """Generate mock status data for demo"""
        return {
            "is_connected": st.session_state.is_connected,
            "signal_strength": st.session_state.signal_strength,
            "current_command": st.session_state.current_command,
            "battery_level": st.session_state.battery_level,
            "active_sessions": np.random.randint(1, 5),
            "total_predictions": np.random.randint(1000, 5000),
            "system_uptime": "2h 15m"
        }
    
    def generate_mock_eeg_data(self) -> List[float]:
        """Generate mock EEG data for 8 channels"""
        return [np.random.normal(0, 20e-6) for _ in range(8)]
    
    def simulate_prediction(self) -> Dict:
        """Simulate a prediction result"""
        commands = ['blink', 'focus', 'relax', 'left_motor', 'right_motor']
        return {
            'predicted_class': np.random.choice(commands),
            'confidence_score': np.random.uniform(0.6, 0.95),
            'processing_time_ms': np.random.randint(40, 120),
            'timestamp': datetime.now()
        }

# Initialize data manager
data_manager = BCIDataManager()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Neural Signal Processing Arduino AI System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.25rem;">ECE Senior Project - Brain-Computer Interface</p>', unsafe_allow_html=True)
    
    # Get system status
    status = data_manager.get_system_status()
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <div class="metric-label">Connection Status</div>
            <div class="metric-value" style="color: {'#059669' if status['is_connected'] else '#dc2626'}">
                {'Connected' if status['is_connected'] else 'Disconnected'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if status['is_connected']:
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-label">Signal Strength</div>
                <div class="metric-value" style="color: #3b82f6">
                    {status['signal_strength']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(status['signal_strength'] / 100)
        else:
            st.markdown(f"""
            <div class="status-card" style="opacity: 0.5">
                <div class="metric-label">Signal Strength</div>
                <div class="metric-value" style="color: #6b7280">
                    --
                </div>
                <p style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">No signal</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if status['is_connected']:
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-label">Current Command</div>
                <div class="metric-value" style="color: #7c3aed">
                    {status['current_command']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card" style="opacity: 0.5">
                <div class="metric-label">Current Command</div>
                <div class="metric-value" style="color: #6b7280">
                    None
                </div>
                <p style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">Inactive</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if status['is_connected']:
            battery_color = "#059669" if status['battery_level'] > 50 else "#f59e0b" if status['battery_level'] > 20 else "#dc2626"
            st.markdown(f"""
            <div class="status-card">
                <div class="metric-label">Battery Level</div>
                <div class="metric-value" style="color: {battery_color}">
                    {status['battery_level']:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(status['battery_level'] / 100)
        else:
            st.markdown(f"""
            <div class="status-card" style="opacity: 0.5">
                <div class="metric-label">Battery Level</div>
                <div class="metric-value" style="color: #6b7280">
                    --
                </div>
                <p style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">System offline</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🔧 Hardware", 
        "📡 Signal Processing", 
        "🤖 AI Model", 
        "💻 Arduino Code", 
        "🎯 Applications"
    ])
    
    with tab1:
        show_overview_tab(status)
    
    with tab2:
        show_hardware_tab()
    
    with tab3:
        show_signal_processing_tab()
    
    with tab4:
        show_ai_model_tab()
    
    with tab5:
        show_arduino_code_tab()
    
    with tab6:
        show_applications_tab()
    
    # Control panel
    show_control_panel(status)

def show_overview_tab(status: Dict):
    """Show overview tab content"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        **Objective:** Develop a real-time brain-computer interface that captures EEG signals, 
        processes them using machine learning algorithms on Arduino, and executes 
        tasks based on classified neural patterns.
        
        **Key Features:**
        - Real-time EEG signal acquisition
        - On-device ML classification
        - Wireless communication
        - Multi-device control capability
        - Low-power operation
        """)
    
    with col2:
        st.markdown("### 🏗️ System Architecture")
        
        # Create architecture flow diagram
        fig = go.Figure()
        
        # Define stages
        stages = [
            {"name": "EEG Sensors", "y": 4, "color": "#3b82f6"},
            {"name": "Signal Processing", "y": 3, "color": "#10b981"},
            {"name": "Arduino AI", "y": 2, "color": "#8b5cf6"},
            {"name": "Task Execution", "y": 1, "color": "#f59e0b"}
        ]
        
        for stage in stages:
            fig.add_trace(go.Scatter(
                x=[0.5], y=[stage["y"]], 
                mode='markers+text',
                marker=dict(size=80, color=stage["color"]),
                text=stage["name"],
                textposition="middle center",
                textfont=dict(color="white", size=12),
                showlegend=False
            ))
        
        # Add arrows
        for i in range(len(stages)-1):
            fig.add_trace(go.Scatter(
                x=[0.5, 0.5], y=[stages[i]["y"]-0.3, stages[i+1]["y"]+0.3],
                mode='lines',
                line=dict(color="gray", width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            height=300,
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0.5, 4.5]),
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time data visualization
    if status['is_connected']:
        st.markdown("### 📈 Real-time Data")
        
        # Generate mock EEG data
        if len(st.session_state.eeg_data) < 100:
            new_data = data_manager.generate_mock_eeg_data()
            st.session_state.eeg_data.append({
                'timestamp': datetime.now(),
                'channels': new_data,
                'signal_quality': np.random.uniform(0.7, 0.95)
            })
        
        # Plot EEG channels
        if st.session_state.eeg_data:
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=[f'Channel {i+1}' for i in range(8)],
                vertical_spacing=0.05
            )
            
            recent_data = st.session_state.eeg_data[-50:]  # Last 50 samples
            
            for ch in range(8):
                row = (ch // 2) + 1
                col = (ch % 2) + 1
                
                channel_data = [[d['channels'][ch] for d in recent_data]]
                timestamps = [d['timestamp'] for d in recent_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps, y=channel_data,
                        mode='lines',
                        name=f'Ch{ch+1}',
                        line=dict(color=px.colors.qualitative.Set3[ch])
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=400,
                title_text="EEG Channels (Real-time)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_hardware_tab():
    """Show hardware tab content"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Hardware Components")
        
        components_data = {
            'Component': [
                'Arduino Nano 33 BLE Sense',
                'ADS1299 EEG Frontend',
                'Dry EEG Electrodes (8-channel)',
                '3.7V LiPo Battery',
                'Custom PCB'
            ],
            'Specification': [
                'ARM Cortex-M4, 64KB RAM',
                '8-channel, 24-bit ADC',
                'Ag/AgCl, 8mm diameter',
                '3.7V, 2000mAh',
                '4-layer, FR4'
            ],
            'Quantity': [1, 1, 8, 1, 1],
            'Cost ($)': [30, 45, 25, 15, 20]
        }
        
        df_components = pd.DataFrame(components_data)
        st.dataframe(df_components, use_container_width=True)
        
        st.markdown(f"**Total Cost: ${df_components['Cost ($)'].sum()}**")
    
    with col2:
        st.markdown("### ⚡ Electrical Specifications")
        
        specs_data = {
            'Parameter': [
                'Operating Voltage',
                'Current Draw (Active)',
                'Current Draw (Sleep)',
                'Signal Range',
                'Frequency Response',
                'Sampling Rate',
                'ADC Resolution'
            ],
            'Value': [
                '3.3V DC',
                '150mA',
                '5mA',
                '±200μV',
                '0.5-50 Hz',
                '250 Hz',
                '24-bit (0.3μV LSB)'
            ]
        }
        
        df_specs = pd.DataFrame(specs_data)
        st.dataframe(df_specs, use_container_width=True)
        
        st.markdown("""
        <div class="alert-success">
            <strong>🔒 Safety Note:</strong> All circuits must be isolated from mains power. 
            Use battery operation only for safety when interfacing with human subjects.
        </div>
        """, unsafe_allow_html=True)

def show_signal_processing_tab():
    """Show signal processing tab content"""
    st.markdown("### 📡 Signal Processing Pipeline")
    
    # Processing stages
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Preprocessing
        - High-pass filter (0.5 Hz)
        - Low-pass filter (50 Hz)
        - Notch filter (60 Hz)
        - Baseline correction
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Feature Extraction
        - Power Spectral Density
        - Band Power (Alpha, Beta)
        - Statistical features
        - Time-domain analysis
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Classification
        - Support Vector Machine
        - Feature normalization
        - Real-time prediction
        - Confidence scoring
        """)
    
    # Filter response visualization
    st.markdown("### 🔍 Digital Filter Response")
    
    # Generate filter response
    from scipy import signal as scipy_signal
    
    # Butterworth low-pass filter (50 Hz cutoff, 250 Hz sampling)
    sos = scipy_signal.butter(4, 50, btype='low', fs=250, output='sos')
    w, h = scipy_signal.sosfreqz(sos, worN=2000, fs=250)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=w, y=20 * np.log10(abs(h)),
        mode='lines',
        name='Magnitude Response',
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.update_layout(
        title="Low-pass Filter Response (50 Hz cutoff)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature extraction demo
    st.markdown("### 🎛️ Feature Extraction Demo")
    
    if st.button("Generate Sample Features"):
        # Generate sample EEG data
        fs = 250
        t = np.linspace(0, 2, fs * 2)  # 2 seconds
        
        # Simulate different frequency components
        alpha_wave = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta_wave = np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
        noise = np.random.normal(0, 0.5, len(t))
        
        eeg_signal = alpha_wave + 0.5 * beta_wave + noise
        
        # Calculate features
        from scipy.signal import welch
        freqs, psd = welch(eeg_signal, fs=fs, nperseg=256)
        
        # Band powers
        alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Time domain plot
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=t, y=eeg_signal,
                mode='lines',
                name='EEG Signal',
                line=dict(color='#10b981')
            ))
            fig_time.update_layout(
                title="Time Domain Signal",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude (μV)",
                height=300
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Frequency domain plot
            fig_freq = go.Figure()
            fig_freq.add_trace(go.Scatter(
                x=freqs, y=psd,
                mode='lines',
                fill='tozeroy',
                name='Power Spectral Density',
                line=dict(color='#8b5cf6')
            ))
            fig_freq.update_layout(
                title="Frequency Domain",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power (μV²/Hz)",
                height=300
            )
            fig_freq.update_xaxes(range=[0, 50])
            st.plotly_chart(fig_freq, use_container_width=True)
        
        # Feature summary
        st.markdown(f"""
        **Extracted Features:**
        - Alpha Power (8-13 Hz): {alpha_power:.4f} μV²/Hz
        - Beta Power (13-30 Hz): {beta_power:.4f} μV²/Hz
        - Mean Amplitude: {np.mean(eeg_signal):.4f} μV
        - Standard Deviation: {np.std(eeg_signal):.4f} μV
        """)

def show_ai_model_tab():
    """Show AI model tab content"""
    st.markdown("### 🤖 Machine Learning Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏗️ Model Architecture")
        
        # Network architecture visualization
        layers_data = {
            'Layer': ['Input Layer', 'Hidden Layer 1', 'Hidden Layer 2', 'Output Layer'],
            'Neurons': [16, 8, 4, 5],
            'Activation': ['None', 'ReLU', 'ReLU', 'Softmax']
        }
        
        df_layers = pd.DataFrame(layers_data)
        st.dataframe(df_layers, use_container_width=True)
        
        # Model performance metrics
        st.markdown("#### 📊 Performance Metrics")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Processing Time'],
            'Value': ['87.5%', '0.88', '0.86', '0.87', '45ms']
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Classification Classes")
        
        classes = [
            {'name': 'Eye Blink', 'color': '#ef4444'},
            {'name': 'Focus/Attention', 'color': '#3b82f6'},
            {'name': 'Relaxation', 'color': '#10b981'},
            {'name': 'Left Motor Imagery', 'color': '#f59e0b'},
            {'name': 'Right Motor Imagery', 'color': '#8b5cf6'}
        ]
        
        for cls in classes:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div style="width: 1rem; height: 1rem; background-color: {cls['color']}; border-radius: 50%; margin-right: 0.5rem;"></div>
                <span>{cls['name']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Training progress simulation
    st.markdown("### 📈 Training Progress")
    
    if st.button("Simulate Training"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate training epochs
        epochs = 50
        accuracies = []
        losses = []
        
        for epoch in range(epochs):
            # Simulate training metrics
            acc = 0.5 + 0.4 * (1 - np.exp(-epoch/10)) + np.random.normal(0, 0.02)
            loss = 2.0 * np.exp(-epoch/15) + np.random.normal(0, 0.05)
            
            accuracies.append(max(0, min(1, acc)))
            losses.append(max(0, loss))
            
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{epochs} - Accuracy: {acc:.3f} - Loss: {loss:.3f}')
            
            time.sleep(0.1)  # Simulate training time
        
        # Plot training curves
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Training Accuracy', 'Training Loss']
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, epochs+1)), y=accuracies,
                mode='lines',
                name='Accuracy',
                line=dict(color='#10b981')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, epochs+1)), y=losses,
                mode='lines',
                name='Loss',
                line=dict(color='#ef4444')
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Training completed! Final accuracy: {accuracies[-1]:.3f}")

def show_arduino_code_tab():
    """Show Arduino code tab content"""
    st.markdown("### 💻 Arduino Implementation")
    
    code_sections = {
        "Setup & Initialization": '''
// Neural Signal Processing Arduino AI System
#include <ArduinoBLE.h>
#include <SPI.h>

// Hardware Configuration
#define ADS1299_CS_PIN 10
#define ADS1299_DRDY_PIN 9
#define SAMPLING_RATE 250
#define CHANNELS 8

// Global Variables
float eegBuffer[CHANNELS][BUFFER_SIZE];
int bufferIndex = 0;

void setup() {
    Serial.begin(115200);
    
    // Initialize SPI for ADS1299
    SPI.begin();
    pinMode(ADS1299_CS_PIN, OUTPUT);
    pinMode(ADS1299_DRDY_PIN, INPUT);
    
    // Initialize EEG frontend
    initializeADS1299();
    
    Serial.println("Neural BCI System Ready");
}
        ''',
        
        "Signal Processing": '''
void readEEGData() {
    digitalWrite(ADS1299_CS_PIN, LOW);
    
    // Read 8 channels (3 bytes each)
    for (int ch = 0; ch < CHANNELS; ch++) {
        long channelData = 0;
        channelData |= ((long)SPI.transfer(0x00) << 16);
        channelData |= ((long)SPI.transfer(0x00) << 8);
        channelData |= SPI.transfer(0x00);
        
        // Convert to voltage
        float voltage = (channelData * 2.4) / 16777216.0;
        
        // Apply filtering
        voltage = highPassFilter(voltage, ch);
        eegBuffer[ch][bufferIndex] = voltage;
    }
    
    digitalWrite(ADS1299_CS_PIN, HIGH);
    bufferIndex++;
}
        ''',
        
        "Neural Network Inference": '''
int runNeuralNetwork() {
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output[OUTPUT_SIZE];
    
    // Layer 1: Input to Hidden1
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = bias1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += scaledFeatures[j] * weights1[j][i];
        }
        hidden1[i] = tanh(hidden1[i]);
    }
    
    // Find maximum output (predicted class)
    int maxIndex = 0;
    float maxValue = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > maxValue) {
            maxValue = output[i];
            maxIndex = i;
        }
    }
    
    return maxIndex;
}
        ''',
        
        "BLE Communication": '''
void sendCommand(int prediction) {
    const char* commands[] = {"BLINK", "FOCUS", "RELAX", "LEFT", "RIGHT"};
    
    if (prediction >= 0 && prediction < OUTPUT_SIZE) {
        commandChar.writeValue(commands[prediction]);
        executeAction(prediction);
        
        Serial.print("Prediction: ");
        Serial.println(commands[prediction]);
    }
}

void executeAction(int action) {
    switch (action) {
        case 0: // BLINK
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
            break;
        case 1: // FOCUS
            Serial.println("Action: Increase focus");
            break;
        // ... other actions
    }
}
        '''
    }
    
    # Code section selector
    section = st.selectbox("Select Code Section:", list(code_sections.keys()))
    
    # Display selected code section
    st.code(code_sections[section], language='cpp')
    
    # Download button for complete code
    if st.button("📥 Download Complete Arduino Code"):
        st.info("Complete Arduino code would be downloaded as neural_bci_arduino.ino")

def show_applications_tab():
    """Show applications tab content"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Practical Applications")
        
        applications = {
            "🦽 Assistive Technology": [
                "Wheelchair control for mobility-impaired users",
                "Computer cursor control via thought",
                "Smart home device automation",
                "Communication aid for speech-impaired individuals"
            ],
            "🎮 Gaming & Entertainment": [
                "Mind-controlled gaming interfaces",
                "Virtual reality navigation",
                "Interactive art installations",
                "Biofeedback training applications"
            ],
            "🏥 Medical & Research": [
                "Neurofeedback therapy systems",
                "Attention and focus training",
                "Sleep quality monitoring",
                "Cognitive load assessment"
            ]
        }
        
        for category, items in applications.items():
            st.markdown(f"#### {category}")
            for item in items:
                st.markdown(f"• {item}")
            st.markdown("")
    
    with col2:
        st.markdown("### 📋 Project Deliverables")
        
        deliverables = {
            "Hardware Deliverables": [
                "✅ Custom PCB design and fabrication",
                "✅ Assembled and tested BCI headset",
                "✅ Arduino-based processing unit",
                "✅ Wireless communication module",
                "✅ Demonstration setup with controlled devices"
            ],
            "Software Deliverables": [
                "✅ Arduino firmware with ML inference",
                "✅ Signal processing algorithms",
                "✅ Training data collection software",
                "✅ Real-time monitoring interface",
                "✅ Calibration and setup utilities"
            ],
            "Documentation": [
                "✅ Technical specification document",
                "✅ Circuit schematics and PCB layouts",
                "✅ Software architecture documentation",
                "✅ User manual and setup guide",
                "✅ Performance evaluation report"
            ]
        }
        
        for category, items in deliverables.items():
            st.markdown(f"#### {category}")
            for item in items:
                st.markdown(item)
            st.markdown("")

def show_control_panel(status: Dict):
    """Show control panel"""
    st.markdown("---")
    st.markdown("### 🎛️ System Control Panel")
    
    if not status['is_connected']:
        st.markdown("""
        <div class="alert-danger">
            <strong>⚠️ System Offline:</strong> Connect the Arduino BCI system to begin neural signal monitoring and control.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-success">
            <strong>✅ System Online:</strong> Real-time control and monitoring interface active.
        </div>
        """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("🔌 Connect" if not status['is_connected'] else "⏹️ Disconnect"):
            st.session_state.is_connected = not st.session_state.is_connected
            if st.session_state.is_connected:
                st.success("System connected successfully!")
                # Start simulation
                st.session_state.signal_strength = np.random.uniform(70, 95)
                commands = ['blink', 'focus', 'relax', 'left_motor', 'right_motor']
                st.session_state.current_command = np.random.choice(commands)
            else:
                st.info("System disconnected")
                st.session_state.signal_strength = 0
                st.session_state.current_command = 'None'
            st.experimental_rerun()
    
    with col2:
        if st.button("🎯 Calibrate Sensors"):
            if status['is_connected']:
                st.info("Calibration started... Please remain still.")
            else:
                st.warning("Connect system first to calibrate sensors")
    
    with col3:
        if st.button("🧠 Start Training"):
            if status['is_connected']:
                st.info("Training session started")
            else:
                st.warning("Connect system first to start training")
    
    with col4:
        if st.button("📊 Export Data"):
            if status['is_connected']:
                st.info("Data export initiated")
            else:
                st.warning("Connect system first to export data")
    
    with col5:
        if st.button("🔧 Diagnostics"):
            if status['is_connected']:
                st.info("Running system diagnostics...")
            else:
                st.warning("Connect system first to run diagnostics")
    
    with col6:
        if st.button("🔄 Refresh"):
            st.experimental_rerun()

# Auto-refresh functionality
def auto_refresh():
    """Auto-refresh the dashboard every 2 seconds if connected"""
    if st.session_state.is_connected:
        # Simulate real-time data updates
        st.session_state.signal_strength = max(0, min(100, 
            st.session_state.signal_strength + np.random.normal(0, 5)))
        
        # Occasionally change command
        if np.random.random() < 0.3:  # 30% chance
            commands = ['blink', 'focus', 'relax', 'left_motor', 'right_motor']
            st.session_state.current_command = np.random.choice(commands)
        
        # Update battery level (slowly decrease)
        st.session_state.battery_level = max(20, 
            st.session_state.battery_level - np.random.uniform(0, 0.1))

# Add auto-refresh mechanism
placeholder = st.empty()
with placeholder.container():
    auto_refresh()

if __name__ == "__main__":
    main()
\`\`\`
