#include <Wire.h>

const int MPU = 0x68; // MPU6050 I2C address
float gyroX, gyroY, gyroZ;
float yaw = 0;
unsigned long lastTime = 0;
float gyroZBias = 0;

void setup() {
  Wire.begin();
  Serial.begin(9600);
  
  // Initialize MPU6050
  Wire.beginTransmission(MPU);
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0);    // Set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  // Calibrate gyroscope
  calibrateGyro();
  
  lastTime = millis();
}

void loop() {
  readGyroData();
  
  unsigned long currentTime = millis();
  float elapsedTime = (currentTime - lastTime) / 1000.0; 
  lastTime = currentTime;
  
  // Integrate the gyroscope data -> yaw angle
  yaw += gyroZ * elapsedTime;
  
  Serial.print("Yaw: ");
  Serial.println(yaw);
  
  delay(100);
}

void readGyroData() {
  Wire.beginTransmission(MPU);
  Wire.write(0x43); // Starting register for Gyro Readings
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 6, true); // Request 6 bytes
  
  gyroX = Wire.read() << 8 | Wire.read(); // X-axis value
  gyroY = Wire.read() << 8 | Wire.read(); // Y-axis value
  gyroZ = Wire.read() << 8 | Wire.read(); // Z-axis value
  
  // Convert to degrees per second
  gyroX /= 131.0;
  gyroY /= 131.0;
  gyroZ /= 131.0;
}

void calibrateGyro() {
  const int numReadings = 100;
  long gyroZSum = 0;
  
  for (int i = 0; i < numReadings; i++) {
    readGyroData();
    gyroZSum += gyroZ;
    delay(10);
  }
  
  gyroZBias = gyroZSum / numReadings;
}