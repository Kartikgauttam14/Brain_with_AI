-- Neural BCI System Database Schema
-- ECE Senior Project - Brain-Computer Interface

-- Create database for storing EEG data and training results
CREATE DATABASE IF NOT EXISTS neural_bci_system;
USE neural_bci_system;

-- Table for storing raw EEG sessions
CREATE TABLE eeg_sessions (
    session_id INT PRIMARY KEY AUTO_INCREMENT,
    subject_id VARCHAR(50) NOT NULL,
    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_type ENUM('training', 'testing', 'calibration') NOT NULL,
    duration_seconds INT NOT NULL,
    sampling_rate INT DEFAULT 250,
    num_channels INT DEFAULT 8,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing raw EEG data points
CREATE TABLE eeg_data (
    data_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    channel_1 FLOAT,
    channel_2 FLOAT,
    channel_3 FLOAT,
    channel_4 FLOAT,
    channel_5 FLOAT,
    channel_6 FLOAT,
    channel_7 FLOAT,
    channel_8 FLOAT,
    signal_quality FLOAT,
    FOREIGN KEY (session_id) REFERENCES eeg_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_timestamp (session_id, timestamp_ms)
);

-- Table for storing extracted features
CREATE TABLE feature_vectors (
    feature_id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    window_start_ms BIGINT NOT NULL,
    window_end_ms BIGINT NOT NULL,
    
    -- Channel 1 features
    ch1_alpha_power FLOAT,
    ch1_beta_power FLOAT,
    ch1_mean_amplitude FLOAT,
    ch1_std_amplitude FLOAT,
    
    -- Channel 2 features
    ch2_alpha_power FLOAT,
    ch2_beta_power FLOAT,
    ch2_mean_amplitude FLOAT,
    ch2_std_amplitude FLOAT,
    
    -- Channel 3 features
    ch3_alpha_power FLOAT,
    ch3_beta_power FLOAT,
    ch3_mean_amplitude FLOAT,
    ch3_std_amplitude FLOAT,
    
    -- Channel 4 features
    ch4_alpha_power FLOAT,
    ch4_beta_power FLOAT,
    ch4_mean_amplitude FLOAT,
    ch4_std_amplitude FLOAT,
    
    -- Ground truth label for training
    true_label ENUM('blink', 'focus', 'relax', 'left_motor', 'right_motor'),
    
    FOREIGN KEY (session_id) REFERENCES eeg_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_window (session_id, window_start_ms)
);

-- Table for storing model training results
CREATE TABLE model_training (
    training_id INT PRIMARY KEY AUTO_INCREMENT,
    model_name VARCHAR(100) NOT NULL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    algorithm_type ENUM('svm', 'neural_network', 'random_forest') NOT NULL,
    
    -- Training parameters
    training_samples INT,
    validation_samples INT,
    test_samples INT,
    
    -- Performance metrics
    training_accuracy FLOAT,
    validation_accuracy FLOAT,
    test_accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    
    -- Model configuration
    hyperparameters JSON,
    feature_importance JSON,
    
    -- Model file path
    model_file_path VARCHAR(255),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing real-time predictions
CREATE TABLE predictions (
    prediction_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    predicted_class ENUM('blink', 'focus', 'relax', 'left_motor', 'right_motor'),
    confidence_score FLOAT,
    processing_time_ms INT,
    
    -- Feature values used for prediction
    feature_vector JSON,
    
    -- Action taken based on prediction
    action_executed VARCHAR(100),
    action_success BOOLEAN,
    
    FOREIGN KEY (session_id) REFERENCES eeg_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session_prediction (session_id, timestamp_ms)
);

-- Table for storing system performance metrics
CREATE TABLE system_performance (
    metric_id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT NOT NULL,
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Hardware metrics
    cpu_usage_percent FLOAT,
    memory_usage_mb FLOAT,
    battery_level_percent FLOAT,
    signal_quality_avg FLOAT,
    
    -- Processing metrics
    avg_processing_time_ms FLOAT,
    max_processing_time_ms FLOAT,
    predictions_per_second FLOAT,
    
    -- Accuracy metrics
    recent_accuracy_percent FLOAT,
    false_positive_rate FLOAT,
    false_negative_rate FLOAT,
    
    FOREIGN KEY (session_id) REFERENCES eeg_sessions(session_id) ON DELETE CASCADE
);

-- Table for storing calibration data
CREATE TABLE calibration_data (
    calibration_id INT PRIMARY KEY AUTO_INCREMENT,
    subject_id VARCHAR(50) NOT NULL,
    calibration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Calibration parameters
    baseline_alpha_power FLOAT,
    baseline_beta_power FLOAT,
    blink_threshold FLOAT,
    focus_threshold FLOAT,
    motor_imagery_threshold FLOAT,
    
    -- Channel-specific calibration
    channel_gains JSON,
    channel_offsets JSON,
    
    -- Validation results
    calibration_accuracy FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_subject_active (subject_id, is_active)
);

-- Create views for common queries
CREATE VIEW session_summary AS
SELECT 
    s.session_id,
    s.subject_id,
    s.session_date,
    s.session_type,
    s.duration_seconds,
    COUNT(DISTINCT f.feature_id) as feature_count,
    COUNT(DISTINCT p.prediction_id) as prediction_count,
    AVG(p.confidence_score) as avg_confidence,
    AVG(sp.recent_accuracy_percent) as avg_accuracy
FROM eeg_sessions s
LEFT JOIN feature_vectors f ON s.session_id = f.session_id
LEFT JOIN predictions p ON s.session_id = p.session_id
LEFT JOIN system_performance sp ON s.session_id = sp.session_id
GROUP BY s.session_id;

-- Create view for model performance comparison
CREATE VIEW model_performance_comparison AS
SELECT 
    model_name,
    algorithm_type,
    training_date,
    test_accuracy,
    precision_score,
    recall_score,
    f1_score,
    training_samples,
    RANK() OVER (ORDER BY test_accuracy DESC) as accuracy_rank
FROM model_training
ORDER BY test_accuracy DESC;

-- Insert sample data for demonstration
INSERT INTO eeg_sessions (subject_id, session_type, duration_seconds, notes) VALUES
('SUBJ001', 'calibration', 300, 'Initial calibration session'),
('SUBJ001', 'training', 600, 'Training session - motor imagery'),
('SUBJ001', 'testing', 180, 'Real-time testing session'),
('SUBJ002', 'calibration', 300, 'Initial calibration session'),
('SUBJ002', 'training', 600, 'Training session - focus/relax');

-- Insert sample model training record
INSERT INTO model_training (
    model_name, 
    algorithm_type, 
    training_samples, 
    validation_samples, 
    test_samples,
    training_accuracy, 
    validation_accuracy, 
    test_accuracy,
    precision_score,
    recall_score,
    f1_score,
    hyperparameters,
    model_file_path
) VALUES (
    'SVM_RBF_v1.0',
    'svm',
    800,
    100,
    100,
    0.95,
    0.88,
    0.87,
    0.88,
    0.86,
    0.87,
    '{"kernel": "rbf", "C": 1.0, "gamma": "scale"}',
    '/models/svm_rbf_v1.0.pkl'
);

-- Create indexes for performance optimization
CREATE INDEX idx_eeg_data_timestamp ON eeg_data(timestamp_ms);
CREATE INDEX idx_feature_vectors_window ON feature_vectors(window_start_ms, window_end_ms);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp_ms);
CREATE INDEX idx_performance_timestamp ON system_performance(metric_timestamp);

-- Create stored procedures for common operations
DELIMITER //

CREATE PROCEDURE GetSessionStatistics(IN p_session_id INT)
BEGIN
    SELECT 
        s.session_id,
        s.subject_id,
        s.session_type,
        s.duration_seconds,
        COUNT(DISTINCT d.data_id) as total_data_points,
        COUNT(DISTINCT f.feature_id) as total_features,
        COUNT(DISTINCT p.prediction_id) as total_predictions,
        AVG(p.confidence_score) as avg_confidence,
        MIN(d.timestamp_ms) as session_start,
        MAX(d.timestamp_ms) as session_end
    FROM eeg_sessions s
    LEFT JOIN eeg_data d ON s.session_id = d.session_id
    LEFT JOIN feature_vectors f ON s.session_id = f.session_id
    LEFT JOIN predictions p ON s.session_id = p.session_id
    WHERE s.session_id = p_session_id
    GROUP BY s.session_id;
END //

CREATE PROCEDURE GetModelAccuracyTrend(IN p_days INT)
BEGIN
    SELECT 
        DATE(training_date) as training_day,
        AVG(test_accuracy) as avg_accuracy,
        COUNT(*) as models_trained,
        MAX(test_accuracy) as best_accuracy
    FROM model_training
    WHERE training_date >= DATE_SUB(CURDATE(), INTERVAL p_days DAY)
    GROUP BY DATE(training_date)
    ORDER BY training_day DESC;
END //

DELIMITER ;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON neural_bci_system.* TO 'arduino_user'@'%';
-- GRANT ALL PRIVILEGES ON neural_bci_system.* TO 'admin_user'@'%';

-- Create triggers for data validation
DELIMITER //

CREATE TRIGGER validate_eeg_data_before_insert
BEFORE INSERT ON eeg_data
FOR EACH ROW
BEGIN
    -- Validate signal quality
    IF NEW.signal_quality < 0 OR NEW.signal_quality > 1 THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Signal quality must be between 0 and 1';
    END IF;
    
    -- Validate channel values (typical EEG range: -200 to +200 microvolts)
    IF ABS(NEW.channel_1) > 200e-6 OR ABS(NEW.channel_2) > 200e-6 OR 
       ABS(NEW.channel_3) > 200e-6 OR ABS(NEW.channel_4) > 200e-6 THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'EEG amplitude out of physiological range';
    END IF;
END //

CREATE TRIGGER update_session_stats_after_prediction
AFTER INSERT ON predictions
FOR EACH ROW
BEGIN
    -- Update session statistics when new prediction is made
    INSERT INTO system_performance (
        session_id, 
        predictions_per_second,
        recent_accuracy_percent
    ) VALUES (
        NEW.session_id,
        1.0, -- Will be calculated properly in real implementation
        NEW.confidence_score * 100
    ) ON DUPLICATE KEY UPDATE
        predictions_per_second = predictions_per_second + 1,
        recent_accuracy_percent = (recent_accuracy_percent + NEW.confidence_score * 100) / 2;
END //

DELIMITER ;

-- Sample queries for data analysis
-- Query 1: Get recent prediction accuracy by class
SELECT 
    predicted_class,
    COUNT(*) as prediction_count,
    AVG(confidence_score) as avg_confidence,
    AVG(CASE WHEN action_success THEN 1 ELSE 0 END) as success_rate
FROM predictions 
WHERE timestamp_ms >= UNIX_TIMESTAMP(DATE_SUB(NOW(), INTERVAL 1 HOUR)) * 1000
GROUP BY predicted_class
ORDER BY prediction_count DESC;

-- Query 2: Find sessions with poor signal quality
SELECT 
    s.session_id,
    s.subject_id,
    s.session_date,
    AVG(d.signal_quality) as avg_signal_quality,
    COUNT(d.data_id) as data_points
FROM eeg_sessions s
JOIN eeg_data d ON s.session_id = d.session_id
GROUP BY s.session_id
HAVING avg_signal_quality < 0.7
ORDER BY avg_signal_quality ASC;

-- Query 3: Model performance over time
SELECT 
    DATE(training_date) as date,
    COUNT(*) as models_trained,
    MAX(test_accuracy) as best_accuracy,
    AVG(test_accuracy) as avg_accuracy
FROM model_training
WHERE training_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY DATE(training_date)
ORDER BY date DESC;

COMMIT;
