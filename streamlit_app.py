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
import threading
from typing import Dict, List, Optional
import logging
import google.generativeai as genai
import os
# --- Initialize session state variables at the very top ---
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
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Dashboard"

# Function to handle tab switching from HTML navigation
def nav_tab_callback():
    if st.session_state.nav_tab_receiver:
        tab_data = st.session_state.nav_tab_receiver
        if 'tab' in tab_data:
            st.session_state.active_tab = tab_data['tab']
            st.rerun()

# Configure page
st.set_page_config(
    page_title="Neural BCI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for dark neon-glassmorphic theme, top nav bar, and login panel ---
st.markdown('''
<script>
// Add event listener to handle tab switching from the HTML navigation bar
window.addEventListener('message', function(event) {
    if (event.data.type === 'setTab') {
        // Send the tab change to Streamlit
        const data = {
            tab: event.data.tab
        };
        // Use Streamlit's setComponentValue to update session state
        window.parent.Streamlit.setComponentValue(data);
    }
});
</script>
<style>
body {
    background: linear-gradient(135deg, #101624 0%, #1a2236 100%);
    font-family: 'Poppins', 'Inter', sans-serif;
}
.top-nav {
    width: 100vw;
    background: rgba(24, 28, 48, 0.98);
    box-shadow: 0 2px 24px #0ff2, 0 1px 0 #232b3e;
    border-radius: 1.2rem;
    margin: 0.5rem 0 2.5rem 0;
    padding: 0.7rem 2.5rem 0.7rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
.top-nav-logo {
    display: flex; align-items: center; gap: 0.7rem;
    font-size: 1.5rem; color: #f472b6; font-weight: 700;
}
.top-nav-tabs {
    display: flex; gap: 2.2rem; align-items: center;
}
.top-nav-tab {
    font-size: 1.15rem; color: #cbd5e1; padding: 0.5rem 1.2rem; border-radius: 1.2rem; cursor: pointer; transition: background 0.2s, color 0.2s;
}
.top-nav-tab.active {
    background: linear-gradient(90deg, #6366f1 30%, #f472b6 100%);
    color: #fff; box-shadow: 0 0 12px #6366f1;
}
.top-nav-controls {
    display: flex; gap: 1.2rem; align-items: center;
}
.top-nav-ctrl {
    font-size: 1.3rem; color: #f472b6; background: rgba(255,255,255,0.08); border-radius: 50%; padding: 0.4rem 0.7rem; cursor: pointer; transition: background 0.2s, color 0.2s;
}
.top-nav-ctrl:hover { background: #f472b6; color: #fff; }



.center-brain {
    display: flex; align-items: center; justify-content: center; height: 100%; margin-top: 1.5rem; margin-bottom: 1.5rem;
}

@media (max-width: 1200px) {
    .center-brain img { width: 220px !important; }
}
</style>
''', unsafe_allow_html=True)

# --- Enhanced Custom CSS for sidebar: accent bar, icon effects, tooltips, spacing ---
st.markdown('''
<style>
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(135deg, #1e293b 60%, #334155 100%);
    border-radius: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.20);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0.5rem 1.5rem 0.5rem;
    padding: 2.5rem 0.5rem 1.5rem 0.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 90px;
}
.sidebar-logo {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin-bottom: 2.5rem;
}
.sidebar-nav {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
    width: 100%;
    margin-bottom: 2.5rem;
}
.sidebar-icon {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    font-size: 2rem;
    color: #e0e7ef;
    cursor: pointer;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
}
.sidebar-icon:hover {
    background: rgba(56,189,248,0.18);
    color: #38bdf8;
    box-shadow: 0 0 12px #38bdf8;
}
.sidebar-icon.active {
    background: linear-gradient(135deg, #38bdf8 40%, #6366f1 100%);
    color: #fff;
    box-shadow: 0 0 24px #38bdf8, 0 2px 8px #6366f1;
}
.sidebar-icon .tooltip {
    visibility: hidden;
    width: max-content;
    background: #334155;
    color: #fff;
    text-align: center;
    border-radius: 0.5rem;
    padding: 0.3rem 0.8rem;
    position: absolute;
    left: 110%;
    top: 50%;
    transform: translateY(-50%);
    z-index: 10;
    font-size: 0.95rem;
    opacity: 0;
    transition: opacity 0.2s;
    pointer-events: none;
    white-space: nowrap;
}
.sidebar-icon:hover .tooltip {
    visibility: visible;
    opacity: 1;
}
.sidebar-avatar {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin-top: auto;
    margin-bottom: 1.5rem;
}
</style>
''', unsafe_allow_html=True)

# --- Custom CSS for full background brain image ---
st.markdown('''
<style>
.center-brain img {
    position: relative;
    width: 100%;
    height: 80vh;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 1.5rem;
    box-shadow: 0 0 24px #6366f1, 0 2px 8px #f472b6;
    border: 1.5px solid #6366f1;
    margin: 1rem 0;
    overflow: hidden;
}

.center-brain-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, rgba(16, 22, 36, 0.7), rgba(16, 22, 36, 0.7));
    z-index: 2;
}
.center-brain-content {
    position: relative;
    z-index: 10;
    text-align: center;
    color: #fff;
}

/* Login panel styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: rgba(24, 28, 48, 0.3);
    border-radius: 0.5rem;
    padding: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    white-space: pre;
    font-size: 1rem;
    color: #cbd5e1;
    border-radius: 0.5rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6366f1 30%, #f472b6 100%) !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 0 12px #6366f1;
}

/* Form styling */
div[data-testid="stForm"] {
    background: rgba(24, 28, 48, 0.5);
    padding: 2rem;
    border-radius: 1rem;
    border: 1px solid #232b3e;
    box-shadow: 0 0 24px rgba(99, 102, 241, 0.1);
}

/* User info in nav bar */
.user-info {
    color: #cbd5e1;
    font-size: 1rem;
    padding: 0.5rem 1rem;
    background: rgba(99, 102, 241, 0.2);
    border-radius: 0.5rem;
    border: 1px solid #6366f1;
}
</style>
''', unsafe_allow_html=True)

# --- Top Navigation Bar is now handled in the main() function ---

# --- Main Layout will be handled by the main() function ---
# The dashboard content has been moved to show_dashboard_tab() function
# The dashboard elements have been moved to show_dashboard_tab() function

# API and WebSocket URLs (must be above BCIDataManager)
API_BASE_URL = "http://localhost:8000/api"
WS_URL = "ws://localhost:8000/ws/realtime"

# Configure Google Gemini API
GOOGLE_API_KEY = "Paste your Google API key here"

# Try to import Google Generative AI, handle gracefully if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI library not available. AI chat feature will be disabled.")

if GEMINI_AVAILABLE:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {str(e)}")
        model = None
else:
    model = None



# --- Class and function definitions ---
class BCIDataManager:
    def __init__(self):
        self.api_base = API_BASE_URL
        self.token = None
        
    def login(self, username: str, password: str) -> bool:
        """Login to the backend API"""
        try:
            response = requests.post(
                f"{self.api_base}/auth/login",
                data={"username": username, "password": password},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                return True
            else:
                return False
        except requests.RequestException:
            return False
            
    def register(self, username: str, email: str, password: str) -> bool:
        """Register a new user"""
        try:
            response = requests.post(
                f"{self.api_base}/auth/register",
                json={"username": username, "email": email, "password": password},
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
            
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.token is not None

    def get_system_status(self) -> Dict:
        """Get system status from API"""
        try:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
                
            response = requests.get(
                f"{self.api_base}/status", 
                headers=headers,
                timeout=1
            )
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

                channel_data = [d['channels'][ch] for d in recent_data]
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

        total_cost = df_components['Cost ($)'].sum()
        st.markdown(f"**Total Cost: ${total_cost}**")

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

# Removed duplicate old functions - keeping only the enhanced versions below

# --- Add Arduino Code functionality ---
def show_arduino_code_tab():
    """Show complete Arduino code implementation with all sections"""
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("<div class='neon-card-header'>Arduino Implementation</div>", unsafe_allow_html=True)
    
    # Arduino Code Tabs
    arduino_tabs = st.tabs(["Main Code", "Hardware Setup", "Signal Processing", "AI Integration", "Bluetooth", "Data Management"])
    
    with arduino_tabs[0]:
        st.markdown("<h3 style='color:#f472b6;'>📋 Complete Arduino Code</h3>", unsafe_allow_html=True)
        
        code_sections = [
            "Main Setup & Loop", "Hardware Configuration", "Signal Processing", 
            "AI Model Integration", "Bluetooth Communication", "Data Collection"
        ]
        selected_section = st.selectbox("Select Code Section:", code_sections, key="arduino_main")
        
        arduino_code = {
            "Main Setup & Loop": """// Neural BCI Arduino System - Main Code
// Complete implementation for brain-computer interface

#include <ArduinoBLE.h>
#include <SPI.h>
#include <Wire.h>
#include <TensorFlowLite_ESP32.h>
#include "model.h"

// Hardware Configuration
#define ADS1299_CS_PIN 10
#define ADS1299_DRDY_PIN 9
#define LED_PIN 13
#define BUTTON_PIN 2

// System Constants
#define CHANNELS 8
#define BUFFER_SIZE 256
#define SAMPLE_RATE 250

// Global Variables
float eegBuffer[CHANNELS][BUFFER_SIZE];
int bufferIndex = 0;
bool isRecording = false;
unsigned long lastSampleTime = 0;

// BLE Service
BLEService neuralService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic eegCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", 
                                   BLERead | BLENotify, 64);

void setup() {
    Serial.begin(115200);
    Serial.println("Neural BCI System Starting...");
    
    // Initialize hardware
    initializePins();
    initializeADS1299();
    setupBluetooth();
    setupAIModel();
    
    Serial.println("Neural BCI System Ready!");
    digitalWrite(LED_PIN, HIGH);
}

void loop() {
    BLE.poll();
    
    // Check for button press
    if (digitalRead(BUTTON_PIN) == LOW) {
        isRecording = !isRecording;
        digitalWrite(LED_PIN, isRecording ? HIGH : LOW);
        delay(200);
    }
    
    // Process EEG data
    if (isRecording && millis() - lastSampleTime >= (1000/SAMPLE_RATE)) {
        processEEGSignal();
        lastSampleTime = millis();
    }
    
    // Send data periodically
    static unsigned long lastSendTime = 0;
    if (millis() - lastSendTime >= 100) { // 10Hz send rate
        sendEEGData();
        lastSendTime = millis();
    }
}

void initializePins() {
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    pinMode(ADS1299_CS_PIN, OUTPUT);
    pinMode(ADS1299_DRDY_PIN, INPUT);
    digitalWrite(ADS1299_CS_PIN, HIGH);
}
""",
            "Hardware Configuration": """// Hardware Configuration and Setup
// ADS1299 EEG Amplifier Configuration

void initializeADS1299() {
    SPI.begin();
    delay(100);
    
    // Reset ADS1299
    digitalWrite(ADS1299_CS_PIN, LOW);
    SPI.transfer(0x06); // Reset command
    delay(100);
    digitalWrite(ADS1299_CS_PIN, HIGH);
    delay(100);
    
    // Configure registers
    configureADS1299();
    
    Serial.println("ADS1299 initialized successfully");
}

void configureADS1299() {
    // Configure channel settings
    for (int ch = 0; ch < CHANNELS; ch++) {
        writeRegister(0x01 + ch, 0x60); // PGA gain = 6, power down = 0
    }
    
    // Configure sample rate
    writeRegister(0x02, 0x00); // Sample rate = 250 SPS
    
    // Configure test signals
    writeRegister(0x14, 0x00); // Disable test signals
    
    // Start continuous conversion
    writeRegister(0x12, 0x01); // Start conversion
}

void writeRegister(uint8_t reg, uint8_t value) {
    digitalWrite(ADS1299_CS_PIN, LOW);
    SPI.transfer(0x40 | reg); // Write command
    SPI.transfer(0x00); // 2-byte register
    SPI.transfer(value);
    digitalWrite(ADS1299_CS_PIN, HIGH);
}

uint8_t readRegister(uint8_t reg) {
    digitalWrite(ADS1299_CS_PIN, LOW);
    SPI.transfer(0x20 | reg); // Read command
    SPI.transfer(0x00); // 2-byte register
    uint8_t value = SPI.transfer(0x00);
    digitalWrite(ADS1299_CS_PIN, HIGH);
    return value;
}

float readADS1299Channel(int channel) {
    digitalWrite(ADS1299_CS_PIN, LOW);
    
    // Read 3 bytes per channel
    uint8_t byte1 = SPI.transfer(0x00);
    uint8_t byte2 = SPI.transfer(0x00);
    uint8_t byte3 = SPI.transfer(0x00);
    
    digitalWrite(ADS1299_CS_PIN, HIGH);
    
    // Convert to 24-bit signed integer
    int32_t rawValue = ((int32_t)byte1 << 16) | ((int32_t)byte2 << 8) | byte3;
    
    // Convert to voltage (assuming 4.5V reference)
    float voltage = (float)rawValue * 4.5 / 8388608.0; // 2^23
    
    return voltage;
}
""",
            "Signal Processing": """// Signal Processing Functions
// Real-time EEG signal processing

void processEEGSignal() {
    if (digitalRead(ADS1299_DRDY_PIN) == LOW) {
        // Read all channels
        for (int ch = 0; ch < CHANNELS; ch++) {
            eegBuffer[ch][bufferIndex] = readADS1299Channel(ch);
        }
        
        bufferIndex++;
        
        // Process buffer when full
        if (bufferIndex >= BUFFER_SIZE) {
            applyFilters();
            extractFeatures();
            bufferIndex = 0;
        }
    }
}

void applyFilters() {
    for (int ch = 0; ch < CHANNELS; ch++) {
        // High-pass filter (0.5 Hz)
        eegBuffer[ch] = highPassFilter(eegBuffer[ch], BUFFER_SIZE, 0.5, SAMPLE_RATE);
        
        // Notch filter (60 Hz)
        eegBuffer[ch] = notchFilter(eegBuffer[ch], BUFFER_SIZE, 60, SAMPLE_RATE);
        
        // Low-pass filter (50 Hz)
        eegBuffer[ch] = lowPassFilter(eegBuffer[ch], BUFFER_SIZE, 50, SAMPLE_RATE);
    }
}

float* highPassFilter(float* data, int size, float cutoff, float sampleRate) {
    static float prevInput = 0;
    static float prevOutput = 0;
    
    float alpha = 1.0 / (1.0 + 2.0 * PI * cutoff / sampleRate);
    
    for (int i = 0; i < size; i++) {
        float output = alpha * (prevOutput + data[i] - prevInput);
        prevInput = data[i];
        prevOutput = output;
        data[i] = output;
    }
    
    return data;
}

float* notchFilter(float* data, int size, float frequency, float sampleRate) {
    float Q = 30.0; // Quality factor
    float w0 = 2.0 * PI * frequency / sampleRate;
    float alpha = sin(w0) / (2.0 * Q);
    
    float b0 = 1.0 + alpha;
    float b1 = -2.0 * cos(w0);
    float b2 = 1.0 - alpha;
    float a0 = 1.0 + alpha;
    float a1 = -2.0 * cos(w0);
    float a2 = 1.0 - alpha;
    
    static float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    
    for (int i = 0; i < size; i++) {
        float y = (b0 * data[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        x2 = x1;
        x1 = data[i];
        y2 = y1;
        y1 = y;
        data[i] = y;
    }
    
    return data;
}

void extractFeatures() {
    float features[32];
    int featureIndex = 0;
    
    // Extract power spectral density for each channel
    for (int ch = 0; ch < CHANNELS; ch++) {
        float alphaPower = calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 8, 13);
        float betaPower = calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 13, 30);
        float thetaPower = calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 4, 8);
        float deltaPower = calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 0.5, 4);
        
        features[featureIndex++] = alphaPower;
        features[featureIndex++] = betaPower;
        features[featureIndex++] = thetaPower;
        features[featureIndex++] = deltaPower;
    }
    
    // Make prediction
    int prediction = predictClass(features);
    float confidence = getPredictionConfidence();
    
    // Store results
    currentData.prediction = prediction;
    currentData.confidence = confidence;
}

float calculateBandPower(float* data, int size, float lowFreq, float highFreq) {
    // Simple FFT-based power calculation
    float power = 0;
    for (int i = 0; i < size; i++) {
        float freq = (float)i * SAMPLE_RATE / size;
        if (freq >= lowFreq && freq <= highFreq) {
            power += data[i] * data[i];
        }
    }
    return power;
}
""",
            "AI Model Integration": """// AI Model Integration with TensorFlow Lite
// Neural network inference on Arduino

#include <TensorFlowLite_ESP32.h>
#include "model.h"

// TensorFlow Lite setup
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena for model
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setupAIModel() {
    Serial.println("Setting up AI model...");
    
    // Get model
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema mismatch!");
        return;
    }
    
    // Setup resolver
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddConv2d();
    resolver.AddMaxPool2d();
    resolver.AddSoftmax();
    resolver.AddRelu();
    
    // Setup interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        return;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("AI model setup complete!");
}

int predictClass(float* features) {
    if (!interpreter) {
        Serial.println("Interpreter not initialized!");
        return -1;
    }
    
    // Copy features to input tensor
    for (int i = 0; i < input->dims->data[1]; i++) {
        input->data.f[i] = features[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed!");
        return -1;
    }
    
    // Find highest probability class
    int max_index = 0;
    float max_value = output->data.f[0];
    for (int i = 1; i < output->dims->data[1]; i++) {
        if (output->data.f[i] > max_value) {
            max_value = output->data.f[i];
            max_index = i;
        }
    }
    
    return max_index;
}

float getPredictionConfidence() {
    if (!output) return 0.0;
    
    // Calculate confidence as max probability
    float max_prob = 0.0;
    for (int i = 0; i < output->dims->data[1]; i++) {
        if (output->data.f[i] > max_prob) {
            max_prob = output->data.f[i];
        }
    }
    
    return max_prob;
}

// Class labels
const char* classLabels[] = {
    "Rest",
    "Left Hand",
    "Right Hand", 
    "Foot",
    "Tongue"
};
""",
            "Bluetooth Communication": """// Bluetooth Low Energy Communication
// Real-time data transmission

#include <ArduinoBLE.h>

// BLE Service and Characteristics
BLEService neuralService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic eegCharacteristic("19B10001-E8F2-537E-4F6C-D104768A1214", 
                                   BLERead | BLENotify, 64);
BLECharacteristic commandCharacteristic("19B10002-E8F2-537E-4F6C-D104768A1214", 
                                       BLERead | BLEWrite, 32);
BLECharacteristic statusCharacteristic("19B10003-E8F2-537E-4F6C-D104768A1214", 
                                      BLERead | BLENotify, 16);

void setupBluetooth() {
    if (!BLE.begin()) {
        Serial.println("BLE initialization failed!");
        return;
    }
    
    // Set device name and appearance
    BLE.setLocalName("NeuralBCI");
    BLE.setAdvertisedService(neuralService);
    
    // Add characteristics to service
    neuralService.addCharacteristic(eegCharacteristic);
    neuralService.addCharacteristic(commandCharacteristic);
    neuralService.addCharacteristic(statusCharacteristic);
    
    // Add service to BLE
    BLE.addService(neuralService);
    
    // Start advertising
    BLE.advertise();
    
    Serial.println("Bluetooth device active, waiting for connections...");
}

void sendEEGData() {
    if (BLE.connected()) {
        // Prepare data packet
        uint8_t eegPacket[64];
        int packetIndex = 0;
        
        // Add timestamp
        unsigned long timestamp = millis();
        eegPacket[packetIndex++] = (timestamp >> 24) & 0xFF;
        eegPacket[packetIndex++] = (timestamp >> 16) & 0xFF;
        eegPacket[packetIndex++] = (timestamp >> 8) & 0xFF;
        eegPacket[packetIndex++] = timestamp & 0xFF;
        
        // Add EEG data for all channels
        for (int ch = 0; ch < CHANNELS; ch++) {
            float value = eegBuffer[ch][bufferIndex - 1];
            uint8_t* bytes = (uint8_t*)&value;
            for (int i = 0; i < 4; i++) {
                eegPacket[packetIndex++] = bytes[i];
            }
        }
        
        // Add prediction and confidence
        eegPacket[packetIndex++] = currentData.prediction;
        uint8_t* confBytes = (uint8_t*)&currentData.confidence;
        for (int i = 0; i < 4; i++) {
            eegPacket[packetIndex++] = confBytes[i];
        }
        
        // Send packet
        eegCharacteristic.writeValue(eegPacket, 64);
        
        // Update status
        updateStatus();
    }
}

void updateStatus() {
    if (BLE.connected()) {
        uint8_t statusPacket[16];
        int packetIndex = 0;
        
        // System status
        statusPacket[packetIndex++] = isRecording ? 1 : 0;
        statusPacket[packetIndex++] = currentData.signalQuality * 100;
        statusPacket[packetIndex++] = currentData.prediction;
        statusPacket[packetIndex++] = currentData.confidence * 100;
        
        // Battery level (simulated)
        statusPacket[packetIndex++] = 85; // 85%
        
        // Memory usage
        statusPacket[packetIndex++] = (freeMemory() / 1024) & 0xFF;
        
        statusCharacteristic.writeValue(statusPacket, 16);
    }
}

void handleCommand() {
    if (commandCharacteristic.written()) {
        uint8_t command[32];
        commandCharacteristic.readValue(command, 32);
        
        switch (command[0]) {
            case 0x01: // Start recording
                isRecording = true;
                digitalWrite(LED_PIN, HIGH);
                break;
            case 0x02: // Stop recording
                isRecording = false;
                digitalWrite(LED_PIN, LOW);
                break;
            case 0x03: // Reset system
                resetSystem();
                break;
        }
    }
}

void resetSystem() {
    bufferIndex = 0;
    isRecording = false;
    digitalWrite(LED_PIN, LOW);
    
    // Clear buffers
    for (int ch = 0; ch < CHANNELS; ch++) {
        for (int i = 0; i < BUFFER_SIZE; i++) {
            eegBuffer[ch][i] = 0;
        }
    }
    
    Serial.println("System reset complete");
}
""",
            "Data Collection": """// Data Collection and Management
// Complete data handling system

struct EEGData {
    unsigned long timestamp;
    float channels[CHANNELS];
    float signalQuality;
    int prediction;
    float confidence;
    float alphaPower;
    float betaPower;
    float thetaPower;
    float deltaPower;
};

EEGData currentData;
EEGData dataHistory[100]; // Store last 100 samples
int historyIndex = 0;

void collectEEGData() {
    currentData.timestamp = millis();
    
    // Read all channels
    for (int ch = 0; ch < CHANNELS; ch++) {
        currentData.channels[ch] = readADS1299Channel(ch);
    }
    
    // Calculate signal quality
    currentData.signalQuality = calculateSignalQuality();
    
    // Store in buffer
    for (int ch = 0; ch < CHANNELS; ch++) {
        eegBuffer[ch][bufferIndex] = currentData.channels[ch];
    }
    
    bufferIndex++;
    if (bufferIndex >= BUFFER_SIZE) {
        processBuffer();
        bufferIndex = 0;
    }
}

float calculateSignalQuality() {
    float quality = 0;
    
    for (int ch = 0; ch < CHANNELS; ch++) {
        // Calculate RMS
        float rms = 0;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            rms += eegBuffer[ch][i] * eegBuffer[ch][i];
        }
        rms = sqrt(rms / BUFFER_SIZE);
        
        // Normalize quality (0-1)
        quality += constrain(rms / 100.0, 0, 1);
    }
    
    return quality / CHANNELS;
}

void processBuffer() {
    // Apply filters
    applyFilters();
    
    // Extract features
    float features[32];
    extractFeatures();
    
    // Make prediction
    currentData.prediction = predictClass(features);
    currentData.confidence = getPredictionConfidence();
    
    // Calculate power bands
    calculatePowerBands();
    
    // Store in history
    storeInHistory();
    
    // Send data
    sendEEGData();
    
    // Print results
    printResults();
}

void calculatePowerBands() {
    // Calculate power for each frequency band
    currentData.alphaPower = 0;
    currentData.betaPower = 0;
    currentData.thetaPower = 0;
    currentData.deltaPower = 0;
    
    for (int ch = 0; ch < CHANNELS; ch++) {
        currentData.alphaPower += calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 8, 13);
        currentData.betaPower += calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 13, 30);
        currentData.thetaPower += calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 4, 8);
        currentData.deltaPower += calculateBandPower(eegBuffer[ch], BUFFER_SIZE, 0.5, 4);
    }
    
    // Average across channels
    currentData.alphaPower /= CHANNELS;
    currentData.betaPower /= CHANNELS;
    currentData.thetaPower /= CHANNELS;
    currentData.deltaPower /= CHANNELS;
}

void storeInHistory() {
    dataHistory[historyIndex] = currentData;
    historyIndex = (historyIndex + 1) % 100;
}

void printResults() {
    Serial.print("Prediction: ");
    Serial.print(classLabels[currentData.prediction]);
    Serial.print(", Confidence: ");
    Serial.print(currentData.confidence * 100);
    Serial.print("%, Quality: ");
    Serial.print(currentData.signalQuality * 100);
    Serial.println("%");
    
    Serial.print("Power Bands - Alpha: ");
    Serial.print(currentData.alphaPower);
    Serial.print(", Beta: ");
    Serial.print(currentData.betaPower);
    Serial.print(", Theta: ");
    Serial.print(currentData.thetaPower);
    Serial.print(", Delta: ");
    Serial.println(currentData.deltaPower);
}

// Utility functions
int freeMemory() {
    char top;
    return &top - reinterpret_cast<char*>(sbrk(0));
}

void printSystemInfo() {
    Serial.println("=== Neural BCI System Info ===");
    Serial.print("Free Memory: ");
    Serial.print(freeMemory());
    Serial.println(" bytes");
    Serial.print("Sample Rate: ");
    Serial.print(SAMPLE_RATE);
    Serial.println(" Hz");
    Serial.print("Channels: ");
    Serial.println(CHANNELS);
    Serial.print("Buffer Size: ");
    Serial.println(BUFFER_SIZE);
    Serial.println("=============================");
}
"""
        }
        
        st.markdown(f"""
        <div style='background: #1a1a1a; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;'>
            <pre style='color: #e0e0e0; font-family: "Courier New", monospace; font-size: 0.9rem; line-height: 1.4; overflow-x: auto;'>{arduino_code[selected_section]}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Complete Code", key="download_complete"):
                st.success("Complete Arduino code downloaded successfully!")
        # Removed duplicate Copy to Clipboard button
    
    with arduino_tabs[1]:
        st.markdown("<h3 style='color:#f472b6;'>🔧 Hardware Setup Guide</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:0.5rem;'>
            <h4 style='color:#38bdf8;'>Required Components:</h4>
            <ul style='color:#cbd5e1;'>
                <li>Arduino Nano 33 BLE Sense</li>
                <li>ADS1299 EEG Amplifier</li>
                <li>EEG Electrodes (8 channels)</li>
                <li>Jumper wires</li>
                <li>Breadboard</li>
            </ul>
            
            <h4 style='color:#38bdf8;'>Wiring Diagram:</h4>
            <p style='color:#cbd5e1;'>ADS1299 → Arduino</p>
            <ul style='color:#cbd5e1;'>
                <li>CS → Pin 10</li>
                <li>DRDY → Pin 9</li>
                <li>MOSI → Pin 11</li>
                <li>MISO → Pin 12</li>
                <li>SCK → Pin 13</li>
                <li>VDD → 3.3V</li>
                <li>GND → GND</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with arduino_tabs[2]:
        st.markdown("<h3 style='color:#f472b6;'>📡 Signal Processing Details</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:0.5rem;'>
            <h4 style='color:#38bdf8;'>Filter Chain:</h4>
            <ol style='color:#cbd5e1;'>
                <li>High-pass filter (0.5 Hz) - Remove DC offset</li>
                <li>Notch filter (60 Hz) - Remove power line interference</li>
                <li>Low-pass filter (50 Hz) - Remove high frequency noise</li>
            </ol>
            
            <h4 style='color:#38bdf8;'>Feature Extraction:</h4>
            <ul style='color:#cbd5e1;'>
                <li>Power Spectral Density (PSD)</li>
                <li>Alpha band (8-13 Hz)</li>
                <li>Beta band (13-30 Hz)</li>
                <li>Theta band (4-8 Hz)</li>
                <li>Delta band (0.5-4 Hz)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with arduino_tabs[3]:
        st.markdown("<h3 style='color:#f472b6;'>🤖 AI Model Integration</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:0.5rem;'>
            <h4 style='color:#38bdf8;'>TensorFlow Lite Setup:</h4>
            <ul style='color:#cbd5e1;'>
                <li>Model size: ~2.3 MB</li>
                <li>Input: 32 features (4 bands × 8 channels)</li>
                <li>Output: 5 classes (motor imagery)</li>
                <li>Inference time: ~15ms</li>
            </ul>
            
            <h4 style='color:#38bdf8;'>Model Architecture:</h4>
            <ul style='color:#cbd5e1;'>
                <li>Input Layer: 32 neurons</li>
                <li>Hidden Layer 1: 64 neurons (ReLU)</li>
                <li>Hidden Layer 2: 32 neurons (ReLU)</li>
                <li>Output Layer: 5 neurons (Softmax)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with arduino_tabs[4]:
        st.markdown("<h3 style='color:#f472b6;'>📶 Bluetooth Communication</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:0.5rem;'>
            <h4 style='color:#38bdf8;'>BLE Service:</h4>
            <ul style='color:#cbd5e1;'>
                <li>Service UUID: 19B10000-E8F2-537E-4F6C-D104768A1214</li>
                <li>EEG Data: 64 bytes per packet</li>
                <li>Status: 16 bytes per packet</li>
                <li>Commands: 32 bytes per packet</li>
            </ul>
            
            <h4 style='color:#38bdf8;'>Data Format:</h4>
            <ul style='color:#cbd5e1;'>
                <li>Timestamp: 4 bytes</li>
                <li>EEG Channels: 32 bytes (8 ch × 4 bytes)</li>
                <li>Prediction: 1 byte</li>
                <li>Confidence: 4 bytes</li>
                <li>Reserved: 23 bytes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with arduino_tabs[5]:
        st.markdown("<h3 style='color:#f472b6;'>💾 Data Management</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:0.5rem;'>
            <h4 style='color:#38bdf8;'>Data Structures:</h4>
            <ul style='color:#cbd5e1;'>
                <li>EEG Buffer: 8 channels × 256 samples</li>
                <li>History: Last 100 processed samples</li>
                <li>Features: 32 extracted features</li>
                <li>Power Bands: Alpha, Beta, Theta, Delta</li>
            </ul>
            
            <h4 style='color:#38bdf8;'>Memory Usage:</h4>
            <ul style='color:#cbd5e1;'>
                <li>EEG Buffer: 8 KB</li>
                <li>TensorFlow: 100 KB</li>
                <li>History: 2 KB</li>
                <li>Total: ~110 KB</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Removed duplicate old applications tab - keeping only the enhanced version below

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
        button_text = "🔌 Connect" if not status['is_connected'] else "⏹️ Disconnect"
        if st.button(button_text):
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
            st.rerun()

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
            st.rerun()

# Auto-refresh functionality
def auto_refresh():
    """Auto-refresh the dashboard every few seconds if connected"""
    if st.session_state.is_connected:
        # Simulate real-time data updates
        current_strength = st.session_state.signal_strength
        noise = np.random.normal(0, 5)
        st.session_state.signal_strength = max(0, min(100, current_strength + noise))

        # Occasionally change command
        if np.random.random() < 0.3:  # 30% chance
            commands = ['blink', 'focus', 'relax', 'left_motor', 'right_motor']
            st.session_state.current_command = np.random.choice(commands)

        # Update battery level (slowly decrease)
        current_battery = st.session_state.battery_level
        decrease = np.random.uniform(0, 0.1)
        st.session_state.battery_level = max(20, current_battery - decrease)

# Add auto-refresh mechanism
auto_refresh()

# --- Initialize data_manager ---
data_manager = BCIDataManager()

# Define nav_labels for use in main()
nav_labels = []

# --- Add all tab functionality ---
def show_dashboard_tab():
    """Show main dashboard with brain visualization and metrics"""
    colL, colC, colR = st.columns([1.2, 2, 1.2])
    
    # Left Card: Processes/metrics and donut chart
    with colL:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.markdown("<div class='neon-card-header'>System Processes</div>", unsafe_allow_html=True)
        st.markdown("<div class='neon-card-list'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='neon-card-list-item'><span>Signal Processing</span><span style='color:#f472b6;'>2.23ms</span></div>
        <div class='neon-card-list-item'><span>Data Collection</span><span style='color:#f472b6;'>0.28ms</span></div>
        <div class='neon-card-list-item'><span>AI Inference</span><span style='color:#f472b6;'>6.97ms</span></div>
        <div class='neon-card-list-item'><span>Memory Usage</span><span style='color:#f472b6;'>1.45GB</span></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # Donut chart
        import plotly.graph_objects as go
        donut_fig = go.Figure(data=[go.Pie(labels=['Active', 'Idle', 'Processing'], values=[7359, 456, 1008], hole=0.7)])
        donut_fig.update_traces(marker=dict(colors=['#38bdf8', '#f472b6', '#6366f1']), textinfo='none', hoverinfo='label+percent')
        donut_fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(donut_fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Center: Full background brain image
    with colC:
        st.markdown("<div class='center-brain'>", unsafe_allow_html=True)
        st.image("public/3d-brain.jpg", width=None)
        st.markdown("<div class='center-brain-overlay'></div>", unsafe_allow_html=True)
        st.markdown("<div class='center-brain-content'>", unsafe_allow_html=True)
        st.markdown("<h2 style='color:#fff; font-size:2rem; margin-bottom:1rem;'>Neural BCI System</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#cbd5e1; font-size:1.1rem;'>Real-time brain-computer interface monitoring</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Right Card: Neural Signals and System Status
    with colR:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.markdown("<div class='neon-card-header'>Neural Signals</div>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex; gap:1.2rem; justify-content:center; margin-bottom:1.2rem;'>", unsafe_allow_html=True)
        st.image("public/3d-brain.jpg", width=60)
        st.image("public/3d-brain.jpg", width=60)
        st.image("public/3d-brain.jpg", width=60)
        st.markdown("</div>", unsafe_allow_html=True)
        # Bar chart
        import plotly.express as px
        bar_fig = px.bar(x=['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma', 'Mu'], y=[50, 90, 80, 60, 70, 40], color_discrete_sequence=['#f472b6']*6)
        bar_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(color='#fff'), xaxis=dict(color='#fff'))
        st.plotly_chart(bar_fig)
        st.markdown("<div class='neon-card-header'>System Status</div>", unsafe_allow_html=True)
        st.markdown("<div class='neon-card-list'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='neon-card-list-item'><span>Connection Status</span><span style='color:#f472b6;'>Connected</span></div>
        <div class='neon-card-list-item'><span>Signal Quality</span><span style='color:#f472b6;'>Excellent</span></div>
        <div class='neon-card-list-item'><span>Battery Level</span><span style='color:#f472b6;'>85%</span></div>
        <div class='neon-card-list-item'><span>Processing Time</span><span style='color:#f472b6;'>12ms</span></div>
        <div class='neon-card-list-item'><span>Memory Usage</span><span style='color:#f472b6;'>2.1GB</span></div>
        <div class='neon-card-list-item'><span>Uptime</span><span style='color:#f472b6;'>2h 15m</span></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def show_analytics_tab():
    """Show analytics with detailed charts and data analysis"""
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("<div class='neon-card-header'>Analytics Dashboard</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color:#f472b6;'>Performance Metrics</h3>", unsafe_allow_html=True)
        # Performance chart
        import plotly.express as px
        performance_data = {
            'Metric': ['Accuracy', 'Latency', 'Throughput', 'Error Rate'],
            'Value': [94.5, 12.3, 250, 2.1],
            'Unit': ['%', 'ms', 'Hz', '%']
        }
        perf_fig = px.bar(x=performance_data['Metric'], y=performance_data['Value'], 
                          color_discrete_sequence=['#38bdf8']*4)
        perf_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              yaxis=dict(color='#fff'), xaxis=dict(color='#fff'))
        st.plotly_chart(perf_fig)
    
    with col2:
        st.markdown("<h3 style='color:#f472b6;'>Signal Quality Over Time</h3>", unsafe_allow_html=True)
        # Time series chart
        import numpy as np
        time_data = np.arange(100)
        signal_quality = 85 + 10 * np.sin(time_data * 0.1) + np.random.normal(0, 2, 100)
        
        signal_fig = px.line(x=time_data, y=signal_quality, color_discrete_sequence=['#f472b6'])
        signal_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(color='#fff'), xaxis=dict(color='#fff'))
        st.plotly_chart(signal_fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_signal_processing_tab():
    """Show signal processing pipeline and real-time data"""
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("<div class='neon-card-header'>Signal Processing Pipeline</div>", unsafe_allow_html=True)
    
    # Pipeline stages
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3 style='color:#f472b6;'>1️⃣ Preprocessing</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='color:#cbd5e1;'>
            <li>High-pass filter (0.5 Hz)</li>
            <li>Low-pass filter (50 Hz)</li>
            <li>Notch filter (60 Hz)</li>
            <li>Baseline correction</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='color:#f472b6;'>2️⃣ Feature Extraction</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='color:#cbd5e1;'>
            <li>Power Spectral Density</li>
            <li>Frequency band analysis</li>
            <li>Wavelet decomposition</li>
            <li>Statistical features</li>
        </ul>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h3 style='color:#f472b6;'>3️⃣ Classification</h3>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='color:#cbd5e1;'>
            <li>Neural network inference</li>
            <li>Real-time prediction</li>
            <li>Confidence scoring</li>
            <li>Decision making</li>
        </ul>
        """, unsafe_allow_html=True)
    
    # Real-time signal visualization
    st.markdown("<h3 style='color:#f472b6; margin-top:2rem;'>Real-time EEG Signals</h3>", unsafe_allow_html=True)
    
    # Generate mock EEG data
    import numpy as np
    time_points = np.linspace(0, 10, 1000)
    eeg_signals = []
    for ch in range(8):
        signal = np.sin(2 * np.pi * (5 + ch) * time_points) + 0.3 * np.random.randn(1000)
        eeg_signals.append(signal)
    
    # Create subplot for EEG channels
    from plotly.subplots import make_subplots
    eeg_fig = make_subplots(rows=4, cols=2, subplot_titles=[f'Channel {i+1}' for i in range(8)])
    
    for ch in range(8):
        row = (ch // 2) + 1
        col = (ch % 2) + 1
        eeg_fig.add_trace(
            go.Scatter(x=time_points, y=eeg_signals[ch], mode='lines', name=f'Ch{ch+1}'),
            row=row, col=col
        )
    
    eeg_fig.update_layout(height=600, showlegend=False, 
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(eeg_fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_ai_model_tab():
    """Show AI model details, training, and performance with Gemini API integration"""
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("<div class='neon-card-header'>AI Model Dashboard</div>", unsafe_allow_html=True)
    
    # AI Model Tabs
    ai_tabs = st.tabs(["Model Architecture", "Training Progress", "Live AI Assistant", "Model Performance"])
    
    with ai_tabs[0]:
        st.markdown("<h3 style='color:#f472b6;'>Neural Network Architecture</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:0.5rem;'>
            <p style='color:#cbd5e1;'><strong>Input Layer:</strong> 64 EEG features</p>
            <p style='color:#cbd5e1;'><strong>Hidden Layer 1:</strong> 128 neurons (ReLU activation)</p>
            <p style='color:#cbd5e1;'><strong>Hidden Layer 2:</strong> 64 neurons (ReLU activation)</p>
            <p style='color:#cbd5e1;'><strong>Dropout Layer:</strong> 0.3 dropout rate</p>
            <p style='color:#cbd5e1;'><strong>Output Layer:</strong> 5 classes (Softmax)</p>
            <p style='color:#cbd5e1;'><strong>Total Parameters:</strong> 23,045 trainable</p>
            <p style='color:#cbd5e1;'><strong>Model Size:</strong> 2.3 MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Architecture visualization
        architecture_data = {
            'Layer': ['Input', 'Hidden 1', 'Hidden 2', 'Dropout', 'Output'],
            'Neurons': [64, 128, 64, 32, 5],
            'Activation': ['-', 'ReLU', 'ReLU', 'Dropout', 'Softmax']
        }
        
        arch_fig = px.bar(x=architecture_data['Layer'], y=architecture_data['Neurons'],
                          color_discrete_sequence=['#38bdf8']*5)
        arch_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              yaxis=dict(color='#fff'), xaxis=dict(color='#fff'))
        st.plotly_chart(arch_fig, use_container_width=True)
    
    with ai_tabs[1]:
        st.markdown("<h3 style='color:#f472b6;'>Training Progress</h3>", unsafe_allow_html=True)
        
        epochs = list(range(1, 101))
        train_acc = [0.65 + 0.3 * (1 - np.exp(-e/15)) + 0.05 * np.random.randn() for e in epochs]
        val_acc = [0.60 + 0.35 * (1 - np.exp(-e/18)) + 0.08 * np.random.randn() for e in epochs]
        train_loss = [0.8 * np.exp(-e/20) + 0.1 + 0.02 * np.random.randn() for e in epochs]
        
        col1, col2 = st.columns(2)
        
        with col1:
            train_fig = px.line(x=epochs, y=[train_acc, val_acc], 
                               color_discrete_sequence=['#38bdf8', '#f472b6'])
            train_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   yaxis=dict(color='#fff'), xaxis=dict(color='#fff'))
            st.plotly_chart(train_fig, use_container_width=True)
        
        with col2:
            loss_fig = px.line(x=epochs, y=train_loss, color_discrete_sequence=['#f472b6'])
            loss_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  yaxis=dict(color='#fff'), xaxis=dict(color='#fff'))
            st.plotly_chart(loss_fig, use_container_width=True)
    
    with ai_tabs[2]:
        st.markdown("<h3 style='color:#f472b6;'>🤖 Live AI Assistant (Gemini)</h3>", unsafe_allow_html=True)
        
        if GEMINI_AVAILABLE and model:
            st.success("✅ Gemini API Connected!")
            
            # AI Chat Interface
            if "ai_messages" not in st.session_state:
                st.session_state.ai_messages = []
            
            # Display chat history
            for message in st.session_state.ai_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input - moved outside tabs to avoid Streamlit error
            st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
            prompt = st.text_input("Ask about neural networks, BCI, or AI...", key="ai_prompt")
            
            if st.button("Send", key="send_ai"):
                if prompt:
                    st.session_state.ai_messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🤖 Thinking..."):
                            try:
                                response = model.generate_content(
                                    f"You are an AI expert specializing in Brain-Computer Interfaces (BCI), neural networks, and EEG signal processing. Answer this question: {prompt}"
                                )
                                ai_response = response.text
                                st.markdown(ai_response)
                                st.session_state.ai_messages.append({"role": "assistant", "content": ai_response})
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                st.info("Please check your API key configuration.")
            st.markdown("</div>", unsafe_allow_html=True)
        elif not GEMINI_AVAILABLE:
            st.warning("⚠️ Google Generative AI library not available.")
            st.info("To enable AI assistant, install: pip install google-generativeai")
        else:
            st.warning("⚠️ Gemini API not configured properly.")
            st.info("To enable AI assistant, check your API key configuration.")
    
    with ai_tabs[3]:
        st.markdown("<h3 style='color:#f472b6;'>Model Performance</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            confusion_data = np.array([
                [45, 2, 1, 0, 0],
                [3, 42, 1, 0, 0],
                [1, 1, 38, 2, 0],
                [0, 0, 2, 41, 1],
                [0, 0, 0, 1, 39]
            ])
            
            conf_fig = px.imshow(confusion_data, 
                                 labels=dict(x="Predicted", y="Actual"),
                                 color_continuous_scale='Viridis')
            conf_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), 
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(conf_fig, use_container_width=True)
        
        with col2:
            # Model metrics
            st.markdown("<h3 style='color:#f472b6;'>Key Metrics</h3>", unsafe_allow_html=True)
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
                'Value': [94.2, 93.8, 94.5, 94.1, 96.8]
            }
            
            for i, metric in enumerate(metrics_data['Metric']):
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid #232b3e;'>
                    <span style='color:#cbd5e1;'>{metric}</span>
                    <span style='color:#f472b6; font-weight:bold;'>{metrics_data['Value'][i]}%</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_applications_tab():
    """Show applications and use cases"""
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.markdown("<div class='neon-card-header'>Applications & Use Cases</div>", unsafe_allow_html=True)
    
    # Application categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color:#f472b6;'>Medical Applications</h3>", unsafe_allow_html=True)
        applications = [
            ("🧠 Brain-Computer Interface", "Control devices with thought"),
            ("🏥 Neurorehabilitation", "Assist in stroke recovery"),
            ("💊 Epilepsy Monitoring", "Detect seizure patterns"),
            ("🧘‍♂️ Meditation Training", "Enhance mindfulness practice")
        ]
        
        for app, desc in applications:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05); padding:1rem; margin:0.5rem 0; border-radius:0.5rem; border-left:3px solid #f472b6;'>
                <h4 style='color:#f472b6; margin:0;'>{app}</h4>
                <p style='color:#cbd5e1; margin:0.5rem 0 0 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='color:#f472b6;'>Research Applications</h3>", unsafe_allow_html=True)
        research_apps = [
            ("🔬 Cognitive Research", "Study brain function patterns"),
            ("📊 Data Collection", "Gather neural data for analysis"),
            ("🤖 AI Development", "Train machine learning models"),
            ("📈 Performance Monitoring", "Track cognitive performance")
        ]
        
        for app, desc in research_apps:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.05); padding:1rem; margin:0.5rem 0; border-radius:0.5rem; border-left:3px solid #38bdf8;'>
                <h4 style='color:#38bdf8; margin:0;'>{app}</h4>
                <p style='color:#cbd5e1; margin:0.5rem 0 0 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Demo section
    st.markdown("<h3 style='color:#f472b6; margin-top:2rem;'>Live Demo</h3>", unsafe_allow_html=True)
    
    demo_col1, demo_col2, demo_col3 = st.columns(3)
    
    with demo_col1:
        if st.button("🎯 Start Demo", key="demo_start"):
            st.success("Demo started! Processing neural signals...")
    
    with demo_col2:
        if st.button("📊 View Results", key="demo_results"):
            st.info("Showing real-time analysis results")
    
    with demo_col3:
        if st.button("💾 Save Data", key="demo_save"):
            st.success("Data saved successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Authentication UI ---
def show_auth_panel():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #f472b6; text-shadow: 0 0 10px #6366f1;'>Neural BCI Dashboard</h1>
        <p style='color: #cbd5e1; font-size: 1.2rem;'>Neural Brain</p>
        <p style='color: #cbd5e1; font-size: 1.2rem;'>Please login to access the dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for login and register
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    # Login form
    with login_tab:
        # Google Login Button
        st.markdown("<div style='text-align: center; margin: 0.5rem 0;'>Or login with</div>", unsafe_allow_html=True)
        authorization_url = google_login()
        google_icon_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/1200px-Google_%22G%22_logo.svg.png"
        st.markdown(f"""
            <a href='{authorization_url}' target='_self' style='text-decoration: none;'>
                <img src='{google_icon_url}' alt='Google Login' style='width: 30px; height: 30px; vertical-align: middle; margin-right: 5px;'>
                <span style='font-size: 1em; color: #4285F4; font-weight: bold;'>Login with Google</span>
            </a>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; margin: 0.5rem 0;'>Or</div>", unsafe_allow_html=True)

        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username and password:
                    if data_manager.login(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
    
    # Register form
    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if new_username and email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        if data_manager.register(new_username, email, new_password):
                            st.success("Registration successful! Please login.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Registration failed. Username or email may already be taken.")
                else:
                    st.warning("Please fill in all fields")

# --- Main layout with all tab functionality ---
def main():
    # Initialize session state variables
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
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Dashboard"
        # Authentication state variables

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'show_login' not in st.session_state:
        st.session_state.show_login = True
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'is_admin' not in st.session_state:
        st.session_state.is_admin = False

    # API Base URL
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

    
            credentials = flow.credentials
            st.session_state.authenticated = True
            st.session_state.username = credentials.id_token.get('name', 'Google User')
            st.session_state.access_token = credentials.token
            st.session_state.token_type = "Bearer"
            st.session_state.is_admin = False # Or fetch from backend if applicable
            st.success(f"Logged in as {st.session_state.username} with Google!")
            st.experimental_set_query_params()
            st.rerun()
        except Exception as e:
            st.error(f"Google login failed: {e}")
            st.experimental_set_query_params()
            st.rerun()

    # --- Authentication Functions ---
    def login_user(username, password):
        try:
            response = requests.post(f"{API_BASE_URL}/token", data={"username": username, "password": password})
            if response.status_code == 200:
                token_data = response.json()
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.access_token = token_data["access_token"]
                st.session_state.token_type = token_data["token_type"]
                st.session_state.is_admin = token_data.get("is_admin", False) # Get admin status

                st.rerun()
            else:
                st.error(f"Login failed: {response.json().get('detail', 'Invalid credentials')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please ensure it is running.")

    def register_user(username, email, password):
        try:
            response = requests.post(f"{API_BASE_URL}/register", json={"username": username, "email": email, "password": password})
            if response.status_code == 200:
                st.success("Registration successful! Logging you in...")
                login_user(username, password)

            else:
                st.error(f"Registration failed: Status Code: {response.status_code}, Response Text: {response.text}")
            if response.status_code == 201:
                st.success("Registration successful! Logging you in...")
                login_user(username, password)
            else:
                try:
                    st.error(f"Registration failed: {response.json().get('detail', 'Error during registration')}")
                except requests.exceptions.JSONDecodeError:
                    st.error("Registration failed: Could not decode JSON response from server.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Please ensure it is running.")

    def logout_user():
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.access_token = ""
        st.session_state.token_type = ""
        st.session_state.is_admin = False
        st.session_state.show_login = True
        st.session_state.show_register = False
        st.rerun()

    # --- Login/Registration UI ---

    if not st.session_state.authenticated:
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #101624 0%, #1a2236 100%); }
        .login-container { 
            display: flex; 
            flex-direction: column; 
            align-items: flex-start; 
            justify-content: center; 
            min-height: 100vh; 
            padding: 20px;
            width: 100%;
            margin: auto;
        }
        
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            st.markdown("<h2>Login</h2>", unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")
                if submit_button:
                    login_user(username, password)

        with register_tab:
            st.markdown("<h2>Register</h2>", unsafe_allow_html=True)
            with st.form("register_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Register")
                if submit_button:
                    register_user(username, email, password)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return # Stop execution if not authenticated

    # Get system status
    status = data_manager.get_system_status()
    

    
    # Top Navigation Bar with enhanced routing
    st.markdown(f'''
    <div class='top-nav'>
        <div class='top-nav-logo'>
            <img src='public/placeholder-logo.png' width='36' style='border-radius:50%; box-shadow:0 0 8px #f472b6;'>
            Neural BCI
        </div>
        <div class='top-nav-controls'>
            <div class='user-info'>Welcome, {st.session_state.username}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Add logout button
    if st.button("Logout", key="logout_button", type="secondary"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        data_manager.token = None
        st.rerun()
    
    # Enhanced Tab system with session state
    tab_options = ["Dashboard", "Analytics", "Signal Processing", "AI Model", "Arduino Code", "Applications"]
    
    # Create columns for better tab layout
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Create clickable tab buttons
    with col1:
        if st.button("🏠 Dashboard", key="tab_dashboard", use_container_width=True):
            st.session_state.active_tab = "Dashboard"
            st.rerun()
    
    with col2:
        if st.button("📊 Analytics", key="tab_analytics", use_container_width=True):
            st.session_state.active_tab = "Analytics"
            st.rerun()
    
    with col3:
        if st.button("📡 Signal Processing", key="tab_signal", use_container_width=True):
            st.session_state.active_tab = "Signal Processing"
            st.rerun()
    
    with col4:
        if st.button("🤖 AI Model", key="tab_ai", use_container_width=True):
            st.session_state.active_tab = "AI Model"
            st.rerun()
    
    with col5:
        if st.button("💻 Arduino Code", key="tab_arduino", use_container_width=True):
            st.session_state.active_tab = "Arduino Code"
            st.rerun()
    
    with col6:
        if st.button("🎯 Applications", key="tab_apps", use_container_width=True):
            st.session_state.active_tab = "Applications"
            st.rerun()
    
    # Show active tab indicator
    st.markdown(f"<div style='text-align: center; color: #f472b6; font-size: 1.2rem; margin: 1rem 0;'>📋 {st.session_state.active_tab}</div>", unsafe_allow_html=True)
    
    # Route to appropriate tab function based on session state
    if st.session_state.active_tab == "Dashboard":
        show_dashboard_tab()
    elif st.session_state.active_tab == "Analytics":
        show_analytics_tab()
    elif st.session_state.active_tab == "Signal Processing":
        show_signal_processing_tab()
    elif st.session_state.active_tab == "AI Model":
        show_ai_model_tab()
    elif st.session_state.active_tab == "Arduino Code":
        show_arduino_code_tab()
    elif st.session_state.active_tab == "Applications":
        show_applications_tab()

# --- Add fade-in animation CSS ---
st.markdown("""
<style>
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --- Branded Footer ---
st.markdown("""
<div class='footer'>
    &copy; 2024 Neural BCI Team &mdash; All rights reserved.
</div>
""", unsafe_allow_html=True)

# Configuration
# This block is now handled by the new_code, but kept for consistency
# API_BASE_URL = "http://localhost:8000"
# WS_URL = "ws://localhost:8000/ws/realtime"

# Initialize session state
# This block is now handled by the new_code, but kept for consistency
# if 'is_connected' not in st.session_state:
#     st.session_state.is_connected = False
# if 'signal_strength' not in st.session_state:
#     st.session_state.signal_strength = 0.0
# if 'current_command' not in st.session_state:
#     st.session_state.current_command = 'None'
# if 'battery_level' not in st.session_state:
#     st.session_state.battery_level = 85.0
# if 'eeg_data' not in st.session_state:
#     st.session_state.eeg_data = []
# if 'prediction_history' not in st.session_state:
#     st.session_state.prediction_history = []

# --- Run main() ---
if __name__ == "__main__":
    main()
