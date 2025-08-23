from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
import logging
import uvicorn
from contextlib import asynccontextmanager
import websockets
import threading
import time
import pickle
from sklearn.svm import SVC
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import redis
import hashlib
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_URL = "sqlite:///./neural_bci.db"  # Change to PostgreSQL in production
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password Hashing
def get_password_hash(password):
    return pwd_context.hash(password)

# Function to create tables and initial admin user
def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Create admin user if not exists
        if not db.query(User).filter(User.username == "admin").first():
            admin_user = User(
                username="admin",
                email="admin@neuralbci.com",
                hashed_password=get_password_hash("admin123"),
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            logger.info("Admin user created.")
    finally:
        db.close()

# Redis for real-time data caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-here"  # Change in production
ALGORITHM = "HS256"

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class EEGSession(Base):
    __tablename__ = "eeg_sessions"
    
    session_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    session_date = Column(DateTime, default=datetime.utcnow)
    session_type = Column(String)  # training, testing, calibration
    duration_seconds = Column(Integer)
    sampling_rate = Column(Integer, default=250)
    num_channels = Column(Integer, default=8)
    notes = Column(Text)
    is_active = Column(Boolean, default=True)

class EEGData(Base):
    __tablename__ = "eeg_data"
    
    data_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    timestamp_ms = Column(Integer)
    channel_data = Column(JSON)  # Store all 8 channels as JSON
    signal_quality = Column(Float)

class FeatureVector(Base):
    __tablename__ = "feature_vectors"
    
    feature_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    window_start_ms = Column(Integer)
    window_end_ms = Column(Integer)
    features = Column(JSON)  # Store feature array as JSON
    true_label = Column(String)  # blink, focus, relax, left_motor, right_motor

class Prediction(Base):
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    timestamp_ms = Column(Integer)
    predicted_class = Column(String)
    confidence_score = Column(Float)
    processing_time_ms = Column(Integer)
    feature_vector = Column(JSON)
    action_executed = Column(String)
    action_success = Column(Boolean)

class ModelTraining(Base):
    __tablename__ = "model_training"
    
    training_id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    training_date = Column(DateTime, default=datetime.utcnow)
    algorithm_type = Column(String)
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    test_accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    hyperparameters = Column(JSON)
    model_file_path = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

class EEGSessionCreate(BaseModel):
    session_type: str = Field(..., pattern="^(training|testing|calibration)$")
    duration_seconds: int
    notes: Optional[str] = None

class EEGSessionResponse(BaseModel):
    session_id: int
    user_id: int
    session_date: datetime
    session_type: str
    duration_seconds: int
    sampling_rate: int
    num_channels: int
    notes: Optional[str]
    is_active: bool

class EEGDataPoint(BaseModel):
    timestamp_ms: int
    channel_data: List[float] = Field(..., min_items=8, max_items=8)
    signal_quality: float = Field(..., ge=0.0, le=1.0)

class EEGDataBatch(BaseModel):
    session_id: int
    data_points: List[EEGDataPoint]

class PredictionResponse(BaseModel):
    prediction_id: int
    session_id: int
    timestamp_ms: int
    predicted_class: str
    confidence_score: float
    processing_time_ms: int
    action_executed: Optional[str]
    action_success: Optional[bool]

class Token(BaseModel):
    access_token: str
    token_type: str

class SystemStatus(BaseModel):
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # Default token expiry
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: User = Depends(get_current_active_user)):
    if current_user.username != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not an admin user")
    return current_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    init_db()
    logger.info("Database initialized.")
    # Load the model during startup
    try:
        with open('scripts/models/eeg_classifier.pkl', 'rb') as f:
            app.state.eeg_model = pickle.load(f)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error("Model file not found. Please ensure 'scripts/models/eeg_classifier.pkl' exists.")
        # Optionally, create a dummy model or exit
        app.state.eeg_model = None # Or a default model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        app.state.eeg_model = None
    yield
    logger.info("Application shutdown.")

app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# API Endpoints
@app.post("/api/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return UserResponse(id=db_user.id, username=db_user.username, email=db_user.email, is_active=db_user.is_active, created_at=db_user.created_at)

@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me/", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/api/users/", response_model=List[UserResponse])
async def read_users(current_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    return SystemStatus(status="ok", message="System is running normally")

# WebSocket endpoint for real-time EEG data
async def websocket_handler(websocket, path):
    logger.info(f"WebSocket connection established: {websocket.remote_address}")
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            session_id = data.get("session_id")
            eeg_data_points = data.get("data_points")

            if session_id and eeg_data_points:
                # Store data in Redis for real-time access
                redis_client.rpush(f"eeg_session:{session_id}", *[json.dumps(dp) for dp in eeg_data_points])
                logger.info(f"Received {len(eeg_data_points)} EEG data points for session {session_id}")

                # Process data (e.g., feature extraction, prediction)
                # This is a placeholder for your real-time processing logic
                # For example, you might trigger a prediction if enough data is accumulated
                # For now, just echo back a confirmation
                await websocket.send(json.dumps({"status": "received", "session_id": session_id, "count": len(eeg_data_points)}))
            else:
                await websocket.send(json.dumps({"status": "error", "message": "Invalid data format"}))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"WebSocket connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket connection terminated: {websocket.remote_address}")

# Run WebSocket server in a separate thread
def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(websocket_handler, "0.0.0.0", 8765)
    loop.run_until_complete(start_server)
    loop.run_forever()

websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
websocket_thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI backend server.")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)

    is_connected: bool
    signal_strength: float
    current_command: str
    battery_level: float
    active_sessions: int
    total_predictions: int
    system_uptime: str

# Signal Processing Class
class EEGSignalProcessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained ML model"""
        try:
            with open('scripts/models/eeg_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained model found, will train new model")
            self.train_default_model()
    
    def train_default_model(self):
        """Train a default model with synthetic data"""
        # Generate synthetic training data
        X, y = self.generate_synthetic_data(1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM
        self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Default model trained - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        # Save model
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'accuracy': test_score
        }
        with open('scripts/models/eeg_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    def generate_synthetic_data(self, n_samples):
        """Generate synthetic EEG data for training"""
        np.random.seed(42)
        X = []
        y = []
        
        for i in range(n_samples):
            class_label = i % 5
            features = self.generate_class_features(class_label)
            X.append(features)
            y.append(class_label)
        
        return np.array(X), np.array(y)
    
    def generate_class_features(self, class_label):
        """Generate features for specific class"""
        features = []
        
        # 4 channels, 4 features each (alpha, beta, mean, std)
        for ch in range(4):
            if class_label == 0:  # Blink
                alpha_power = np.random.normal(0.02, 0.005)
                beta_power = np.random.normal(0.08, 0.02)
                mean_amp = np.random.normal(0.0001, 0.00002)
                std_amp = np.random.normal(0.00005, 0.00001)
            elif class_label == 1:  # Focus
                alpha_power = np.random.normal(0.01, 0.003)
                beta_power = np.random.normal(0.15, 0.03)
                mean_amp = np.random.normal(0.00005, 0.00001)
                std_amp = np.random.normal(0.00008, 0.00002)
            elif class_label == 2:  # Relax
                alpha_power = np.random.normal(0.05, 0.01)
                beta_power = np.random.normal(0.03, 0.01)
                mean_amp = np.random.normal(0.00003, 0.000005)
                std_amp = np.random.normal(0.00003, 0.000005)
            elif class_label == 3:  # Left motor
                alpha_power = np.random.normal(0.03, 0.008)
                beta_power = np.random.normal(0.06, 0.015)
                mean_amp = np.random.normal(0.00004, 0.00001)
                std_amp = np.random.normal(0.00006, 0.00001)
            else:  # Right motor
                alpha_power = np.random.normal(0.025, 0.006)
                beta_power = np.random.normal(0.07, 0.018)
                mean_amp = np.random.normal(0.00006, 0.00001)
                std_amp = np.random.normal(0.00004, 0.000008)
            
            features.extend([alpha_power, beta_power, mean_amp, std_amp])
        
        return features
    
    def apply_filters(self, data, channel):
        """Apply digital filters to EEG data"""
        # High-pass filter (0.5 Hz)
        sos_hp = signal.butter(4, 0.5, btype='high', fs=self.sampling_rate, output='sos')
        data = signal.sosfilt(sos_hp, data)
        
        # Low-pass filter (50 Hz)
        sos_lp = signal.butter(4, 50, btype='low', fs=self.sampling_rate, output='sos')
        data = signal.sosfilt(sos_lp, data)
        
        # Notch filter (60 Hz)
        sos_notch = signal.iirnotch(60, 30, fs=self.sampling_rate)
        data = signal.sosfilt(sos_notch, data)
        
        return data
    
    def extract_features(self, eeg_window):
        """Extract features from EEG window"""
        features = []
        
        for ch in range(min(4, len(eeg_window))):
            channel_data = np.array(eeg_window[ch])
            
            # Apply filters
            filtered_data = self.apply_filters(channel_data, ch)
            
            # Time domain features
            mean_amp = np.mean(filtered_data)
            std_amp = np.std(filtered_data)
            
            # Frequency domain features
            freqs, psd = signal.welch(filtered_data, fs=self.sampling_rate, nperseg=min(256, len(filtered_data)))
            
            # Alpha band power (8-13 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.mean(psd[alpha_mask]) if np.any(alpha_mask) else 0
            
            # Beta band power (13-30 Hz)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            beta_power = np.mean(psd[beta_mask]) if np.any(beta_mask) else 0
            
            features.extend([alpha_power, beta_power, mean_amp, std_amp])
        
        return np.array(features)
    
    def predict(self, features):
        """Make prediction using trained model"""
        if self.model is None:
            return 0, 0.0
        
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled)[0])
        
        return int(prediction), float(confidence)

# Initialize signal processor
signal_processor = EEGSignalProcessor()

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.device_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_type: str = "web"):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_type == "arduino":
            self.device_connections["arduino"] = websocket
        logger.info(f"New {client_type} connection established")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Remove from device connections
        for key, conn in list(self.device_connections.items()):
            if conn == websocket:
                del self.device_connections[key]
                break
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Neural BCI Backend starting up...")
    # Create models directory
    import os
    os.makedirs("models", exist_ok=True)
    yield
    # Shutdown
    logger.info("Neural BCI Backend shutting down...")

app = FastAPI(
    title="Neural BCI Backend API",
    description="RESTful API for Neural Signal Processing Arduino AI System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Allow Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models for API
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    is_admin: bool = False

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Utility functions for authentication
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not pwd_context.verify(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(db: Session = Depends(get_db), token: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_current_admin_user(current_user: User = Depends(get_current_active_user)):
    if not current_user.username == "admin": # Simple admin check
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not an admin")
    return current_user

@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized and admin user checked.")



# Authentication endpoints
@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/api/auth/login")
async def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# EEG Session endpoints
@app.post("/api/sessions", response_model=EEGSessionResponse)
async def create_session(
    session: EEGSessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_session = EEGSession(
        user_id=current_user.id,
        session_type=session.session_type,
        duration_seconds=session.duration_seconds,
        notes=session.notes
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    
    # Cache session info in Redis
    redis_client.setex(
        f"session:{db_session.session_id}",
        3600,  # 1 hour expiry
        json.dumps({
            "session_id": db_session.session_id,
            "user_id": current_user.id,
            "session_type": session.session_type,
            "start_time": datetime.utcnow().isoformat()
        })
    )
    
    return db_session

@app.get("/api/sessions", response_model=List[EEGSessionResponse])
async def get_sessions(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sessions = db.query(EEGSession).filter(
        EEGSession.user_id == current_user.id
    ).offset(skip).limit(limit).all()
    return sessions

@app.get("/api/sessions/{session_id}", response_model=EEGSessionResponse)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EEGSession).filter(
        EEGSession.session_id == session_id,
        EEGSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    session = db.query(EEGSession).filter(
        EEGSession.session_id == session_id,
        EEGSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.is_active = False
    db.commit()
    
    # Remove from Redis cache
    redis_client.delete(f"session:{session_id}")
    
    return {"message": "Session deleted successfully"}

# EEG Data endpoints
@app.post("/api/data/batch")
async def upload_eeg_batch(
    batch: EEGDataBatch,
    db: Session = Depends(get_db)
):
    # Verify session exists
    session = db.query(EEGSession).filter(
        EEGSession.session_id == batch.session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Process and store data points
    processed_count = 0
    for data_point in batch.data_points:
        db_data = EEGData(
            session_id=batch.session_id,
            timestamp_ms=data_point.timestamp_ms,
            channel_data=data_point.channel_data,
            signal_quality=data_point.signal_quality
        )
        db.add(db_data)
        processed_count += 1
        
        # Cache latest data point in Redis for real-time access
        redis_client.setex(
            f"latest_data:{batch.session_id}",
            60,  # 1 minute expiry
            json.dumps({
                "timestamp_ms": data_point.timestamp_ms,
                "channel_data": data_point.channel_data,
                "signal_quality": data_point.signal_quality
            })
        )
    
    db.commit()
    
    # Broadcast to connected clients
    await manager.broadcast(json.dumps({
        "type": "eeg_data",
        "session_id": batch.session_id,
        "data_points": len(batch.data_points),
        "latest_quality": batch.data_points[-1].signal_quality
    }))
    
    return {"message": f"Processed {processed_count} data points"}

@app.get("/api/data/{session_id}")
async def get_eeg_data(
    session_id: int,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify session ownership
    session = db.query(EEGSession).filter(
        EEGSession.session_id == session_id,
        EEGSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    query = db.query(EEGData).filter(EEGData.session_id == session_id)
    
    if start_time:
        query = query.filter(EEGData.timestamp_ms >= start_time)
    if end_time:
        query = query.filter(EEGData.timestamp_ms <= end_time)
    
    data = query.order_by(EEGData.timestamp_ms.desc()).limit(limit).all()
    
    return {
        "session_id": session_id,
        "data_points": len(data),
        "data": [
            {
                "timestamp_ms": d.timestamp_ms,
                "channel_data": d.channel_data,
                "signal_quality": d.signal_quality
            }
            for d in data
        ]
    }

# Real-time prediction endpoint
@app.post("/api/predict")
async def make_prediction(
    session_id: int,
    eeg_window: List[List[float]],  # 8 channels x window_size samples
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        # Extract features
        features = signal_processor.extract_features(eeg_window)
        
        # Make prediction
        predicted_class, confidence = signal_processor.predict(features)
        
        # Map class index to label
        class_labels = ["blink", "focus", "relax", "left_motor", "right_motor"]
        predicted_label = class_labels[predicted_class]
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Store prediction
        db_prediction = Prediction(
            session_id=session_id,
            timestamp_ms=int(time.time() * 1000),
            predicted_class=predicted_label,
            confidence_score=confidence,
            processing_time_ms=processing_time,
            feature_vector=features.tolist()
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        # Cache latest prediction
        redis_client.setex(
            f"latest_prediction:{session_id}",
            60,
            json.dumps({
                "predicted_class": predicted_label,
                "confidence_score": confidence,
                "timestamp_ms": db_prediction.timestamp_ms
            })
        )
        
        # Broadcast prediction to connected clients
        await manager.broadcast(json.dumps({
            "type": "prediction",
            "session_id": session_id,
            "predicted_class": predicted_label,
            "confidence_score": confidence,
            "processing_time_ms": processing_time
        }))
        
        return {
            "prediction_id": db_prediction.prediction_id,
            "predicted_class": predicted_label,
            "confidence_score": confidence,
            "processing_time_ms": processing_time,
            "features": features.tolist()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Model training endpoint
@app.post("/api/train")
async def train_model(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    def train_model_task():
        try:
            # Get training data from database
            feature_data = db.query(FeatureVector).filter(
                FeatureVector.true_label.isnot(None)
            ).all()
            
            if len(feature_data) < 50:
                logger.warning("Insufficient training data, using synthetic data")
                X, y = signal_processor.generate_synthetic_data(1000)
            else:
                X = np.array([f.features for f in feature_data])
                y = np.array([f.true_label for f in feature_data])
                # Convert labels to integers
                label_map = {"blink": 0, "focus": 1, "relax": 2, "left_motor": 3, "right_motor": 4}
                y = np.array([label_map.get(label, 0) for label in y])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Save model
            model_data = {
                'model': model,
                'scaler': scaler,
                'accuracy': test_score
            }
            model_path = f"models/eeg_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Update global model
            signal_processor.model = model
            signal_processor.scaler = scaler
            
            # Store training record
            training_record = ModelTraining(
                model_name=f"SVM_RBF_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                algorithm_type="svm",
                training_accuracy=train_score,
                test_accuracy=test_score,
                model_file_path=model_path,
                hyperparameters={"kernel": "rbf", "C": 1.0, "gamma": "scale"}
            )
            db.add(training_record)
            db.commit()
            
            logger.info(f"Model trained successfully - Test accuracy: {test_score:.3f}")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
    
    background_tasks.add_task(train_model_task)
    return {"message": "Model training started in background"}

# System status endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status(db: Session = Depends(get_db)):
    # Check if Arduino is connected
    is_connected = "arduino" in manager.device_connections
    
    # Get latest signal data from Redis
    signal_strength = 0.0
    current_command = "None"
    
    if is_connected:
        # Get latest prediction
        latest_pred = redis_client.get("latest_prediction:*")
        if latest_pred:
            pred_data = json.loads(latest_pred)
            current_command = pred_data.get("predicted_class", "None")
            signal_strength = pred_data.get("confidence_score", 0.0) * 100
    
    # Get active sessions count
    active_sessions = db.query(EEGSession).filter(EEGSession.is_active == True).count()
    
    # Get total predictions count
    total_predictions = db.query(Prediction).count()
    
    return SystemStatus(
        is_connected=is_connected,
        signal_strength=signal_strength,
        current_command=current_command,
        battery_level=85.0,  # This would come from Arduino
        active_sessions=active_sessions,
        total_predictions=total_predictions,
        system_uptime="2h 15m"  # This would be calculated from startup time
    )

# WebSocket endpoints
@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket, "web")
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - could implement specific message handling
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/arduino")
async def arduino_websocket(websocket: WebSocket):
    await manager.connect(websocket, "arduino")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                # Handle different message types from Arduino
                if message.get("type") == "eeg_data":
                    # Process EEG data
                    await handle_arduino_eeg_data(message)
                elif message.get("type") == "status":
                    # Update system status
                    await handle_arduino_status(message)
                
                # Broadcast to web clients
                await manager.broadcast(data)
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON received from Arduino")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Arduino disconnected")

async def handle_arduino_eeg_data(message):
    """Handle EEG data from Arduino"""
    try:
        session_id = message.get("session_id", 1)  # Default session
        eeg_data = message.get("data", [])
        
        if len(eeg_data) >= 8:  # Ensure we have all 8 channels
            # Make prediction if we have enough data
            if len(eeg_data[0]) >= 125:  # 0.5 second window
                features = signal_processor.extract_features(eeg_data)
                predicted_class, confidence = signal_processor.predict(features)
                
                class_labels = ["BLINK", "FOCUS", "RELAX", "LEFT", "RIGHT"]
                command = class_labels[predicted_class]
                
                # Send command back to Arduino
                if "arduino" in manager.device_connections:
                    await manager.device_connections["arduino"].send_text(
                        json.dumps({
                            "type": "command",
                            "command": command,
                            "confidence": confidence
                        })
                    )
                
    except Exception as e:
        logger.error(f"Error handling Arduino EEG data: {str(e)}")

async def handle_arduino_status(message):
    """Handle status updates from Arduino"""
    try:
        # Cache status in Redis
        redis_client.setex(
            "arduino_status",
            30,  # 30 second expiry
            json.dumps(message)
        )
    except Exception as e:
        logger.error(f"Error handling Arduino status: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Analytics endpoints
@app.get("/api/analytics/session/{session_id}")
async def get_session_analytics(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify session ownership
    session = db.query(EEGSession).filter(
        EEGSession.session_id == session_id,
        EEGSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get predictions for this session
    predictions = db.query(Prediction).filter(
        Prediction.session_id == session_id
    ).all()
    
    # Calculate analytics
    total_predictions = len(predictions)
    if total_predictions == 0:
        return {"message": "No predictions found for this session"}
    
    # Class distribution
    class_counts = {}
    confidence_scores = []
    processing_times = []
    
    for pred in predictions:
        class_counts[pred.predicted_class] = class_counts.get(pred.predicted_class, 0) + 1
        confidence_scores.append(pred.confidence_score)
        processing_times.append(pred.processing_time_ms)
    
    return {
        "session_id": session_id,
        "total_predictions": total_predictions,
        "class_distribution": class_counts,
        "average_confidence": np.mean(confidence_scores),
        "average_processing_time_ms": np.mean(processing_times),
        "max_processing_time_ms": max(processing_times),
        "min_processing_time_ms": min(processing_times)
    }

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI backend server.")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the FastAPI server on.")
    args = parser.parse_args()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=args.port,
        reload=True,
        log_level="info"
    )
