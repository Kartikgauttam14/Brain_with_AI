import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./neural_bci.db"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # EEG Processing
    sampling_rate: int = 250
    num_channels: int = 8
    window_size_ms: int = 500
    
    # Model
    model_path: str = "models/"
    default_model_name: str = "eeg_classifier.pkl"
    
    # Arduino Communication
    arduino_timeout: int = 30
    max_reconnect_attempts: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()
