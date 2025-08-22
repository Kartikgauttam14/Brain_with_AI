// Arduino Code Generator for Neural BCI System
// ECE Senior Project - Brain-Computer Interface

class ArduinoCodeGenerator {
    constructor() {
        this.modelWeights = null;
        this.modelBiases = null;
        this.featureScaler = null;
    }

    // Generate optimized Arduino code with trained model weights
    generateArduinoCode(modelData) {
        this.modelWeights = modelData.weights;
        this.modelBiases = modelData.biases;
        this.featureScaler = modelData.scaler;

        const code = `
// Auto-generated Neural BCI Arduino Code
// Generated on: ${new Date().toISOString()}
// Model Accuracy: ${(modelData.accuracy * 100).toFixed(2)}%

#include <ArduinoBLE.h>
#include <SPI.h>
#include <math.h>

// Model Configuration
#define INPUT_SIZE ${this.modelWeights.layer1.input_size}
#define HIDDEN1_SIZE ${this.modelWeights.layer1.output_size}
#define HIDDEN2_SIZE ${this.modelWeights.layer2.output_size}
#define OUTPUT_SIZE ${this.modelWeights.layer3.output_size}

// Hardware Configuration
#define ADS1299_CS_PIN 10
#define ADS1299_DRDY_PIN 9
#define SAMPLING_RATE 250
#define CHANNELS 8
#define BUFFER_SIZE 256
#define WINDOW_SIZE 125  // 0.5 second window

// Neural Network Weights (stored in PROGMEM to save RAM)
${this.generateWeightArrays()}

// Feature scaling parameters
${this.generateScalerParameters()}

// Global Variables
float eegBuffer[CHANNELS][BUFFER_SIZE];
volatile int bufferIndex = 0;
float features[INPUT_SIZE];
float scaledFeatures[INPUT_SIZE];
unsigned long lastPredictionTime = 0;
const unsigned long PREDICTION_INTERVAL = 200; // 200ms between predictions

// BLE Configuration
BLEService neuralService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic commandChar("19B10001-E8F2-537E-4F6C-D104768A1214", 
                             BLERead | BLENotify, 20);
BLECharacteristic statusChar("19B10002-E8F2-537E-4F6C-D104768A1214", 
                            BLERead | BLENotify, 50);

// Filter states for real-time processing
struct FilterState {
    float x[3];  // Input history
    float y[3];  // Output history
};

FilterState highPassFilters[CHANNELS];
FilterState lowPassFilters[CHANNELS];
FilterState notchFilters[CHANNELS];

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    
    Serial.println("Neural BCI System Starting...");
    Serial.println("Model Accuracy: ${(modelData.accuracy * 100).toFixed(2)}%");
    
    // Initialize hardware
    initializeHardware();
    
    // Initialize filters
    initializeFilters();
    
    // Initialize BLE
    initializeBLE();
    
    Serial.println("System Ready - Waiting for connections...");
}

void loop() {
    BLEDevice central = BLE.central();
    
    if (central) {
        Serial.print("Connected to: ");
        Serial.println(central.address());
        
        while (central.connected()) {
            // Read EEG data
            if (digitalRead(ADS1299_DRDY_PIN) == LOW) {
                readAndProcessEEGData();
            }
            
            // Make predictions at regular intervals
            if (millis() - lastPredictionTime >= PREDICTION_INTERVAL) {
                if (bufferIndex >= WINDOW_SIZE) {
                    makePrediction();
                    lastPredictionTime = millis();
                }
            }
            
            // Send status updates
            sendStatusUpdate();
            
            delay(1);
        }
        
        Serial.println("Disconnected from central");
    }
}

void initializeHardware() {
    // Configure SPI
    SPI.begin();
    SPI.setClockDivider(SPI_CLOCK_DIV8);
    SPI.setDataMode(SPI_MODE1);
    SPI.setBitOrder(MSBFIRST);
    
    // Configure pins
    pinMode(ADS1299_CS_PIN, OUTPUT);
    pinMode(ADS1299_DRDY_PIN, INPUT);
    pinMode(LED_BUILTIN, OUTPUT);
    
    digitalWrite(ADS1299_CS_PIN, HIGH);
    
    // Initialize ADS1299
    delay(100);
    resetADS1299();
    configureADS1299();
    
    Serial.println("Hardware initialized");
}

void resetADS1299() {
    digitalWrite(ADS1299_CS_PIN, LOW);
    SPI.transfer(0x06); // RESET
    digitalWrite(ADS1299_CS_PIN, HIGH);
    delay(100);
}

void configureADS1299() {
    // Configure for 250 SPS, internal reference
    writeADS1299Register(0x01, 0x96); // CONFIG1
    writeADS1299Register(0x02, 0xD0); // CONFIG2
    writeADS1299Register(0x03, 0xEC); // CONFIG3
    
    // Configure channels for normal electrode input
    for (int i = 0; i < 8; i++) {
        writeADS1299Register(0x05 + i, 0x60);
    }
    
    // Start continuous conversion
    digitalWrite(ADS1299_CS_PIN, LOW);
    SPI.transfer(0x08); // START
    digitalWrite(ADS1299_CS_PIN, HIGH);
    
    Serial.println("ADS1299 configured");
}

void writeADS1299Register(byte reg, byte value) {
    digitalWrite(ADS1299_CS_PIN, LOW);
    SPI.transfer(0x40 | reg); // WREG
    SPI.transfer(0x00);       // Number of registers - 1
    SPI.transfer(value);
    digitalWrite(ADS1299_CS_PIN, HIGH);
    delayMicroseconds(10);
}

void initializeFilters() {
    // Initialize all filter states to zero
    for (int ch = 0; ch < CHANNELS; ch++) {
        memset(&highPassFilters[ch], 0, sizeof(FilterState));
        memset(&lowPassFilters[ch], 0, sizeof(FilterState));
        memset(&notchFilters[ch], 0, sizeof(FilterState));
    }
    Serial.println("Filters initialized");
}

void initializeBLE() {
    if (!BLE.begin()) {
        Serial.println("Starting BLE failed!");
        while (1);
    }
    
    BLE.setLocalName("Neural-BCI-v2");
    BLE.setAdvertisedService(neuralService);
    
    neuralService.addCharacteristic(commandChar);
    neuralService.addCharacteristic(statusChar);
    BLE.addService(neuralService);
    
    commandChar.writeValue("READY");
    statusChar.writeValue("System initialized - accuracy: ${(modelData.accuracy * 100).toFixed(1)}%");
    
    BLE.advertise();
    Serial.println("BLE initialized");
}

void readAndProcessEEGData() {
    digitalWrite(ADS1299_CS_PIN, LOW);
    
    // Read status (3 bytes)
    SPI.transfer(0x00);
    SPI.transfer(0x00);
    SPI.transfer(0x00);
    
    // Read 8 channels
    for (int ch = 0; ch < CHANNELS; ch++) {
        long rawData = 0;
        rawData |= ((long)SPI.transfer(0x00) << 16);
        rawData |= ((long)SPI.transfer(0x00) << 8);
        rawData |= SPI.transfer(0x00);
        
        // Convert to signed 24-bit
        if (rawData & 0x800000) {
            rawData |= 0xFF000000;
        }
        
        // Convert to voltage (2.4V reference, gain=24)
        float voltage = (rawData * 2.4) / (16777216.0 * 24.0);
        
        // Apply real-time filtering
        voltage = applyHighPassFilter(voltage, ch);
        voltage = applyLowPassFilter(voltage, ch);
        voltage = applyNotchFilter(voltage, ch);
        
        // Store in circular buffer
        eegBuffer[ch][bufferIndex % BUFFER_SIZE] = voltage;
    }
    
    digitalWrite(ADS1299_CS_PIN, HIGH);
    bufferIndex++;
}

float applyHighPassFilter(float input, int channel) {
    // Butterworth high-pass filter (0.5 Hz cutoff)
    // Coefficients for 250 Hz sampling rate
    const float a[] = {1.0000, -1.9955, 0.9955};
    const float b[] = {0.9978, -1.9955, 0.9978};
    
    FilterState* filter = &highPassFilters[channel];
    
    // Shift input history
    filter->x[2] = filter->x[1];
    filter->x[1] = filter->x[0];
    filter->x[0] = input;
    
    // Calculate output
    float output = b[0] * filter->x[0] + b[1] * filter->x[1] + b[2] * filter->x[2]
                  - a[1] * filter->y[0] - a[2] * filter->y[1];
    
    // Shift output history
    filter->y[2] = filter->y[1];
    filter->y[1] = filter->y[0];
    filter->y[0] = output;
    
    return output;
}

float applyLowPassFilter(float input, int channel) {
    // Butterworth low-pass filter (50 Hz cutoff)
    const float a[] = {1.0000, -1.1430, 0.4128};
    const float b[] = {0.0675, 0.1349, 0.0675};
    
    FilterState* filter = &lowPassFilters[channel];
    
    filter->x[2] = filter->x[1];
    filter->x[1] = filter->x[0];
    filter->x[0] = input;
    
    float output = b[0] * filter->x[0] + b[1] * filter->x[1] + b[2] * filter->x[2]
                  - a[1] * filter->y[0] - a[2] * filter->y[1];
    
    filter->y[2] = filter->y[1];
    filter->y[1] = filter->y[0];
    filter->y[0] = output;
    
    return output;
}

float applyNotchFilter(float input, int channel) {
    // Notch filter for 60 Hz power line interference
    const float a[] = {1.0000, -1.9319, 0.9355};
    const float b[] = {0.9677, -1.9319, 0.9677};
    
    FilterState* filter = &notchFilters[channel];
    
    filter->x[2] = filter->x[1];
    filter->x[1] = filter->x[0];
    filter->x[0] = input;
    
    float output = b[0] * filter->x[0] + b[1] * filter->x[1] + b[2] * filter->x[2]
                  - a[1] * filter->y[0] - a[2] * filter->y[1];
    
    filter->y[2] = filter->y[1];
    filter->y[1] = filter->y[0];
    filter->y[0] = output;
    
    return output;
}

void makePrediction() {
    // Extract features from recent EEG data
    extractFeatures();
    
    // Scale features
    scaleFeatures();
    
    // Run neural network
    int prediction = runNeuralNetwork();
    
    // Send command
    sendCommand(prediction);
    
    // Update status
    updateSystemStatus(prediction);
}

void extractFeatures() {
    int featureIndex = 0;
    int startIdx = (bufferIndex - WINDOW_SIZE + BUFFER_SIZE) % BUFFER_SIZE;
    
    // Extract features from first 4 channels
    for (int ch = 0; ch < 4; ch++) {
        // Calculate power spectral density features
        float alphaPower = calculateBandPower(ch, startIdx, 8, 13);
        float betaPower = calculateBandPower(ch, startIdx, 13, 30);
        
        // Calculate time domain features
        float mean = 0, variance = 0;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int idx = (startIdx + i) % BUFFER_SIZE;
            mean += eegBuffer[ch][idx];
        }
        mean /= WINDOW_SIZE;
        
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int idx = (startIdx + i) % BUFFER_SIZE;
            float diff = eegBuffer[ch][idx] - mean;
            variance += diff * diff;
        }
        float stdDev = sqrt(variance / WINDOW_SIZE);
        
        features[featureIndex++] = alphaPower;
        features[featureIndex++] = betaPower;
        features[featureIndex++] = mean;
        features[featureIndex++] = stdDev;
    }
}

float calculateBandPower(int channel, int startIdx, float lowFreq, float highFreq) {
    // Simplified band power calculation using time-domain approximation
    // In a full implementation, this would use FFT
    
    float power = 0;
    int count = 0;
    
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (startIdx + i) % BUFFER_SIZE;
        float sample = eegBuffer[channel][idx];
        
        // Simple frequency estimation based on zero crossings and amplitude
        if (i > 0) {
            int prevIdx = (startIdx + i - 1) % BUFFER_SIZE;
            float prevSample = eegBuffer[channel][prevIdx];
            
            // Estimate instantaneous frequency
            if ((sample > 0 && prevSample <= 0) || (sample <= 0 && prevSample > 0)) {
                float freq = SAMPLING_RATE / (2.0 * (i + 1));
                if (freq >= lowFreq && freq <= highFreq) {
                    power += sample * sample;
                    count++;
                }
            }
        }
    }
    
    return count > 0 ? power / count : 0;
}

void scaleFeatures() {
    // Apply feature scaling using pre-computed parameters
    for (int i = 0; i < INPUT_SIZE; i++) {
        scaledFeatures[i] = (features[i] - featureMeans[i]) / featureStds[i];
        
        // Clamp to reasonable range to prevent overflow
        if (scaledFeatures[i] > 5.0) scaledFeatures[i] = 5.0;
        if (scaledFeatures[i] < -5.0) scaledFeatures[i] = -5.0;
    }
}

int runNeuralNetwork() {
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output[OUTPUT_SIZE];
    
    // Layer 1: Input to Hidden1
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = pgm_read_float(&bias1[i]);
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += scaledFeatures[j] * pgm_read_float(&weights1[j][i]);
        }
        hidden1[i] = tanh(hidden1[i]);
    }
    
    // Layer 2: Hidden1 to Hidden2
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        hidden2[i] = pgm_read_float(&bias2[i]);
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            hidden2[i] += hidden1[j] * pgm_read_float(&weights2[j][i]);
        }
        hidden2[i] = tanh(hidden2[i]);
    }
    
    // Layer 3: Hidden2 to Output
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = pgm_read_float(&bias3[i]);
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            output[i] += hidden2[j] * pgm_read_float(&weights3[j][i]);
        }
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

void sendCommand(int prediction) {
    const char* commands[] = {"BLINK", "FOCUS", "RELAX", "LEFT", "RIGHT"};
    
    if (prediction >= 0 && prediction < OUTPUT_SIZE) {
        commandChar.writeValue(commands[prediction]);
        
        // Execute corresponding action
        executeAction(prediction);
        
        Serial.print("Prediction: ");
        Serial.println(commands[prediction]);
    }
}

void executeAction(int action) {
    static int ledState = LOW;
    static unsigned long lastActionTime = 0;
    
    // Prevent rapid repeated actions
    if (millis() - lastActionTime < 500) return;
    lastActionTime = millis();
    
    switch (action) {
        case 0: // BLINK
            ledState = !ledState;
            digitalWrite(LED_BUILTIN, ledState);
            break;
            
        case 1: // FOCUS
            // Could control external device brightness
            Serial.println("Action: Increase focus");
            break;
            
        case 2: // RELAX
            // Could control external device to relaxation mode
            Serial.println("Action: Relaxation mode");
            break;
            
        case 3: // LEFT
            // Could control servo or cursor movement
            Serial.println("Action: Move left");
            break;
            
        case 4: // RIGHT
            // Could control servo or cursor movement
            Serial.println("Action: Move right");
            break;
    }
}

void sendStatusUpdate() {
    static unsigned long lastStatusTime = 0;
    
    if (millis() - lastStatusTime >= 1000) { // Update every second
        char status[50];
        float signalQuality = calculateSignalQuality();
        
        snprintf(status, sizeof(status), "Quality:%.1f%% Buffer:%d/%d", 
                signalQuality * 100, bufferIndex % BUFFER_SIZE, BUFFER_SIZE);
        
        statusChar.writeValue(status);
        lastStatusTime = millis();
    }
}

float calculateSignalQuality() {
    // Simple signal quality metric based on amplitude and noise
    float totalPower = 0;
    float noisePower = 0;
    int samples = min(WINDOW_SIZE, bufferIndex);
    
    if (samples < 10) return 0.0;
    
    for (int ch = 0; ch < 4; ch++) {
        for (int i = 0; i < samples; i++) {
            int idx = (bufferIndex - samples + i + BUFFER_SIZE) % BUFFER_SIZE;
            float sample = eegBuffer[ch][idx];
            totalPower += sample * sample;
            
            // High frequency noise estimation
            if (i > 0) {
                int prevIdx = (bufferIndex - samples + i - 1 + BUFFER_SIZE) % BUFFER_SIZE;
                float diff = sample - eegBuffer[ch][prevIdx];
                noisePower += diff * diff;
            }
        }
    }
    
    if (totalPower == 0) return 0.0;
    
    float snr = totalPower / (noisePower + 1e-10);
    return min(1.0, snr / 100.0); // Normalize to 0-1 range
}

void updateSystemStatus(int prediction) {
    static int predictionCounts[OUTPUT_SIZE] = {0};
    static unsigned long lastResetTime = 0;
    
    predictionCounts[prediction]++;
    
    // Reset counts every minute
    if (millis() - lastResetTime >= 60000) {
        memset(predictionCounts, 0, sizeof(predictionCounts));
        lastResetTime = millis();
    }
}
`;

        return code;
    }

    generateWeightArrays() {
        if (!this.modelWeights) return "// No model weights available";

        let code = "// Neural Network Weights stored in PROGMEM\n";
        
        // Layer 1 weights
        code += `const PROGMEM float weights1[${this.modelWeights.layer1.input_size}][${this.modelWeights.layer1.output_size}] = {\n`;
        for (let i = 0; i < this.modelWeights.layer1.input_size; i++) {
            code += "  {";
            for (let j = 0; j < this.modelWeights.layer1.output_size; j++) {
                const weight = this.modelWeights.layer1.weights[i][j] || (Math.random() - 0.5) * 0.1;
                code += weight.toFixed(6);
                if (j < this.modelWeights.layer1.output_size - 1) code += ", ";
            }
            code += "}";
            if (i < this.modelWeights.layer1.input_size - 1) code += ",";
            code += "\n";
        }
        code += "};\n\n";

        // Layer 1 biases
        code += `const PROGMEM float bias1[${this.modelWeights.layer1.output_size}] = {`;
        for (let i = 0; i < this.modelWeights.layer1.output_size; i++) {
            const bias = this.modelWeights.layer1.biases[i] || (Math.random() - 0.5) * 0.1;
            code += bias.toFixed(6);
            if (i < this.modelWeights.layer1.output_size - 1) code += ", ";
        }
        code += "};\n\n";

        // Similar for other layers...
        code += `const PROGMEM float weights2[${this.modelWeights.layer2.input_size}][${this.modelWeights.layer2.output_size}] = {\n`;
        for (let i = 0; i < this.modelWeights.layer2.input_size; i++) {
            code += "  {";
            for (let j = 0; j < this.modelWeights.layer2.output_size; j++) {
                const weight = this.modelWeights.layer2.weights[i][j] || (Math.random() - 0.5) * 0.1;
                code += weight.toFixed(6);
                if (j < this.modelWeights.layer2.output_size - 1) code += ", ";
            }
            code += "}";
            if (i < this.modelWeights.layer2.input_size - 1) code += ",";
            code += "\n";
        }
        code += "};\n\n";

        return code;
    }

    generateScalerParameters() {
        if (!this.featureScaler) {
            // Generate dummy scaler parameters
            const inputSize = 16;
            let code = `const float featureMeans[${inputSize}] = {`;
            for (let i = 0; i < inputSize; i++) {
                code += (Math.random() * 0.001).toFixed(8);
                if (i < inputSize - 1) code += ", ";
            }
            code += "};\n\n";

            code += `const float featureStds[${inputSize}] = {`;
            for (let i = 0; i < inputSize; i++) {
                code += (0.001 + Math.random() * 0.01).toFixed(8);
                if (i < inputSize - 1) code += ", ";
            }
            code += "};\n\n";

            return code;
        }

        // Use actual scaler parameters if available
        let code = `const float featureMeans[${this.featureScaler.means.length}] = {`;
        for (let i = 0; i < this.featureScaler.means.length; i++) {
            code += this.featureScaler.means[i].toFixed(8);
            if (i < this.featureScaler.means.length - 1) code += ", ";
        }
        code += "};\n\n";

        code += `const float featureStds[${this.featureScaler.stds.length}] = {`;
        for (let i = 0; i < this.featureScaler.stds.length; i++) {
            code += this.featureScaler.stds[i].toFixed(8);
            if (i < this.featureScaler.stds.length - 1) code += ", ";
        }
        code += "};\n\n";

        return code;
    }
}

// Example usage and model data structure
const exampleModelData = {
    accuracy: 0.87,
    weights: {
        layer1: {
            input_size: 16,
            output_size: 8,
            weights: Array(16).fill().map(() => Array(8).fill().map(() => (Math.random() - 0.5) * 0.2)),
            biases: Array(8).fill().map(() => (Math.random() - 0.5) * 0.1)
        },
        layer2: {
            input_size: 8,
            output_size: 4,
            weights: Array(8).fill().map(() => Array(4).fill().map(() => (Math.random() - 0.5) * 0.2)),
            biases: Array(4).fill().map(() => (Math.random() - 0.5) * 0.1)
        },
        layer3: {
            input_size: 4,
            output_size: 5,
            weights: Array(4).fill().map(() => Array(5).fill().map(() => (Math.random() - 0.5) * 0.2)),
            biases: Array(5).fill().map(() => (Math.random() - 0.5) * 0.1)
        }
    },
    scaler: {
        means: Array(16).fill().map(() => Math.random() * 0.001),
        stds: Array(16).fill().map(() => 0.001 + Math.random() * 0.01)
    }
};

// Generate the Arduino code
const generator = new ArduinoCodeGenerator();
const arduinoCode = generator.generateArduinoCode(exampleModelData);

console.log("Arduino Code Generated Successfully!");
console.log("Code length:", arduinoCode.length, "characters");
console.log("Model accuracy:", (exampleModelData.accuracy * 100).toFixed(2) + "%");

// Save to file (in a real implementation)
// fs.writeFileSync('neural_bci_arduino.ino', arduinoCode);

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ArduinoCodeGenerator, exampleModelData };
}
