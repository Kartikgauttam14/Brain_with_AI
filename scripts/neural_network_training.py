import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Neural Signal Processing and ML Training Script
# ECE Senior Project - Brain-Computer Interface

class EEGSignalProcessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
    def generate_synthetic_eeg_data(self, n_samples=1000, n_channels=8):
        """Generate synthetic EEG data for demonstration"""
        np.random.seed(42)
        
        # Define frequency bands
        alpha_freq = np.random.uniform(8, 13, n_samples)
        beta_freq = np.random.uniform(13, 30, n_samples)
        theta_freq = np.random.uniform(4, 8, n_samples)
        
        data = []
        labels = []
        
        for i in range(n_samples):
            # Generate different signal patterns for each class
            class_label = i % 5  # 5 classes: blink, focus, relax, left, right
            
            if class_label == 0:  # Blink
                signal = self._generate_blink_pattern(n_channels)
            elif class_label == 1:  # Focus
                signal = self._generate_focus_pattern(n_channels, alpha_freq[i])
            elif class_label == 2:  # Relax
                signal = self._generate_relax_pattern(n_channels, alpha_freq[i])
            elif class_label == 3:  # Left motor imagery
                signal = self._generate_motor_imagery_pattern(n_channels, 'left')
            else:  # Right motor imagery
                signal = self._generate_motor_imagery_pattern(n_channels, 'right')
            
            data.append(signal)
            labels.append(class_label)
        
        return np.array(data), np.array(labels)
    
    def _generate_blink_pattern(self, n_channels, duration=1.0):
        """Generate eye blink artifact pattern"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        
        # Strong frontal electrode activity
        signal = np.zeros((n_channels, len(t)))
        
        # Blink artifact - strong in frontal channels
        blink_amplitude = 100e-6  # 100 microvolts
        blink_pattern = blink_amplitude * np.exp(-((t - 0.5) / 0.1) ** 2)
        
        signal[0] = blink_pattern + np.random.normal(0, 5e-6, len(t))  # Fp1
        signal[1] = blink_pattern + np.random.normal(0, 5e-6, len(t))  # Fp2
        
        # Other channels with reduced amplitude
        for ch in range(2, n_channels):
            signal[ch] = 0.3 * blink_pattern + np.random.normal(0, 10e-6, len(t))
        
        return signal
    
    def _generate_focus_pattern(self, n_channels, alpha_freq, duration=1.0):
        """Generate focused attention pattern"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        signal = np.zeros((n_channels, len(t)))
        
        # Reduced alpha activity during focus
        alpha_amplitude = 15e-6
        beta_amplitude = 25e-6  # Increased beta during focus
        
        for ch in range(n_channels):
            # Reduced alpha, increased beta
            alpha_component = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
            beta_component = beta_amplitude * np.sin(2 * np.pi * 20 * t)
            noise = np.random.normal(0, 8e-6, len(t))
            
            signal[ch] = 0.5 * alpha_component + 1.5 * beta_component + noise
        
        return signal
    
    def _generate_relax_pattern(self, n_channels, alpha_freq, duration=1.0):
        """Generate relaxed state pattern"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        signal = np.zeros((n_channels, len(t)))
        
        # Strong alpha activity during relaxation
        alpha_amplitude = 40e-6
        
        for ch in range(n_channels):
            alpha_component = alpha_amplitude * np.sin(2 * np.pi * alpha_freq * t)
            noise = np.random.normal(0, 5e-6, len(t))
            
            signal[ch] = alpha_component + noise
        
        return signal
    
    def _generate_motor_imagery_pattern(self, n_channels, direction, duration=1.0):
        """Generate motor imagery pattern"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        signal = np.zeros((n_channels, len(t)))
        
        # Motor imagery affects sensorimotor rhythms (8-30 Hz)
        mu_freq = 10  # Mu rhythm frequency
        beta_freq = 20
        
        for ch in range(n_channels):
            if direction == 'left':
                # Right hemisphere activity for left motor imagery
                if ch >= n_channels // 2:  # Right hemisphere channels
                    amplitude_factor = 1.5
                else:
                    amplitude_factor = 0.8
            else:  # right
                # Left hemisphere activity for right motor imagery
                if ch < n_channels // 2:  # Left hemisphere channels
                    amplitude_factor = 1.5
                else:
                    amplitude_factor = 0.8
            
            mu_component = 20e-6 * amplitude_factor * np.sin(2 * np.pi * mu_freq * t)
            beta_component = 15e-6 * amplitude_factor * np.sin(2 * np.pi * beta_freq * t)
            noise = np.random.normal(0, 8e-6, len(t))
            
            signal[ch] = mu_component + beta_component + noise
        
        return signal
    
    def extract_features(self, signal):
        """Extract features from EEG signal"""
        features = []
        n_channels = signal.shape[0]
        
        for ch in range(min(4, n_channels)):  # Use first 4 channels
            # Time domain features
            mean_amp = np.mean(signal[ch])
            std_amp = np.std(signal[ch])
            
            # Frequency domain features (simplified)
            fft = np.fft.fft(signal[ch])
            freqs = np.fft.fftfreq(len(signal[ch]), 1/self.sampling_rate)
            
            # Alpha band power (8-13 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.mean(np.abs(fft[alpha_mask])**2)
            
            # Beta band power (13-30 Hz)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            beta_power = np.mean(np.abs(fft[beta_mask])**2)
            
            features.extend([alpha_power, beta_power, mean_amp, std_amp])
        
        return np.array(features)
    
    def train_classifier(self, X, y):
        """Train SVM classifier"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train SVM
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = svm_model.score(X_train_scaled, y_train)
        test_score = svm_model.score(X_test_scaled, y_test)
        
        y_pred = svm_model.predict(X_test_scaled)
        
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Testing Accuracy: {test_score:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Blink', 'Focus', 'Relax', 'Left', 'Right']))
        
        return svm_model, X_test_scaled, y_test, y_pred

def main():
    print("Neural Signal Processing and ML Training")
    print("ECE Senior Project - Brain-Computer Interface")
    print("=" * 50)
    
    # Initialize processor
    processor = EEGSignalProcessor()
    
    # Generate synthetic EEG data
    print("Generating synthetic EEG data...")
    raw_data, labels = processor.generate_synthetic_eeg_data(n_samples=1000)
    
    # Extract features
    print("Extracting features...")
    features = []
    for signal in raw_data:
        feature_vector = processor.extract_features(signal)
        features.append(feature_vector)
    
    features = np.array(features)
    print(f"Feature matrix shape: {features.shape}")
    
    # Train classifier
    print("\nTraining classifier...")
    model, X_test, y_test, y_pred = processor.train_classifier(features, labels)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Neural Signal Classification')
    plt.colorbar()
    
    classes = ['Blink', 'Focus', 'Relax', 'Left', 'Right']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Feature importance analysis
    print("\nFeature Analysis:")
    feature_names = []
    for ch in range(4):
        feature_names.extend([
            f'Ch{ch+1}_Alpha', f'Ch{ch+1}_Beta', 
            f'Ch{ch+1}_Mean', f'Ch{ch+1}_Std'
        ])
    
    # Display feature statistics
    feature_stats = pd.DataFrame({
        'Feature': feature_names,
        'Mean': np.mean(features, axis=0),
        'Std': np.std(features, axis=0)
    })
    
    print(feature_stats)
    
    print("\nTraining completed successfully!")
    print("Model ready for deployment on Arduino")

if __name__ == "__main__":
    main()
