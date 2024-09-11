#include <LiquidCrystal.h>


// Define motor pin
int motorPin = 5;  // PWM pin

// Define encoder pins
int encoderPinA = 2;  
int encoderPinB = 3;  

volatile long pulseCount = 0;  // Total pulse count (shared with ISR)
unsigned long previousMillis = 0;
const float pulsesPerRevolution = 24.0;  // Set this to the PPR of encoder


int setpoint = 100; // Assume target speed
int current_speed = 0;

// PID parameters
float kp = 1.5;  // Proportional gain
float ki = 0.5;  // Integral gain
float kd = 0.05;  // Derivative gain

// Variables for PID calculations
float cum_error = 0;
float previousError = 0;
unsigned long previousTime = 0;

// LCD pin configuration 
LiquidCrystal lcd(13, 12, 11, 10, 9, 8);  

// Exponential Smoothing Filter parameters
float alpha = 0.3;  // Smoothing factor
float smoothed_speed = 0;  // Initialize with 0 or any initial speed

float ExponentialSmoothing(float previousValue, float newValue) {
    return alpha * newValue + (1 - alpha) * previousValue;
}

class PID {
  private:
    float kp, ki, kd;
    float cum_error, previousError;
    unsigned long previousTime;

  public:
    // Constructor
    PID(float p, float i, float d) : kp(p), ki(i), kd(d), cum_error(0), previousError(0), previousTime(0) {}

    // Compute method for PID
    float compute(float setpoint, float actualSpeed) {
        unsigned long currentTime = millis();
        float elapsedTime = (currentTime - previousTime) / 1000.0;
        previousTime = currentTime;

        // Calculate error
        float error = setpoint - actualSpeed;
        cum_error += error * elapsedTime;
        float rate_error = (error - previousError) / elapsedTime;
        previousError = error;

        // PID output
        return kp * error + ki * cum_error + kd * rate_error;
    }
};

// Function to count encoder pulses (ISR)
void countPulse() {
    pulseCount++;
}

// Function to calculate speed in RPM from encoder pulses
float calculateSpeed() {
    unsigned long currentMillis = millis();
    float timeDifference = (currentMillis - previousMillis) / 1000.0;  // Convert to seconds

    // Avoid division by zero in time difference
    if (timeDifference == 0) {
        timeDifference = 0.001;
    }

    // Calculate revolutions based on pulse count and PPR (pulses per revolution)
    float revolutions = pulseCount / pulsesPerRevolution;

    // Calculate RPM (revolutions per minute)
    float rpm = (revolutions / timeDifference) * 60;

    // Reset pulse count and update time
    pulseCount = 0;
    previousMillis = currentMillis;

    return rpm;
}

PID motorPID(kp, ki, kd);  // Create PID instance for motor control

void setup() {
    pinMode(motorPin, OUTPUT);
    // Initialize encoder pins
    pinMode(encoderPinA, INPUT);
    pinMode(encoderPinB, INPUT);

    // Attach interrupt to encoderPinA to count pulses
    attachInterrupt(digitalPinToInterrupt(encoderPinA), countPulse, RISING);

    // Initialize the LCD
    lcd.begin(16, 2);  // 16 columns, 2 rows

    // Print initial message on the LCD
    lcd.setCursor(0, 0);
    lcd.print("Speed Control");

    Serial.begin(9600);
}

void loop() {
    // Calculate current speed using encoder feedback
    current_speed = calculateSpeed();

    // Apply filter using exponential smoothing
    smoothed_speed = ExponentialSmoothing(smoothed_speed, current_speed);

    // Compute PID output
    float output = motorPID.compute(setpoint, current_speed);

    // Adjust motor speed using PWM
    analogWrite(motorPin, constrain(output, 0, 255));  // Constrain PWM to valid range

    // Print the current speed on the LCD
    lcd.setCursor(0, 1);  // Move cursor to the second row
    lcd.print("Speed: ");
    lcd.print(current_speed);  // Print speed
    lcd.print(" RPM");

    // Print the current speed
    Serial.print("Current Speed (RPM): ");
    Serial.println(current_speed);
}