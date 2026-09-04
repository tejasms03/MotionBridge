/*
  MotionBridge ESP32 Firmware
  ---------------------------
  Reads:
    - Wrist  MPU6050  on I2C address 0x68 (AD0 -> GND)
    - Elbow  MPU6050  on I2C address 0x69 (AD0 -> 3.3V)
    - MyoWare 2.0 EMG on GPIO 34 (12-bit ADC, 0-4095)  [optional]

  Sends over UART @ 115200 baud, ~100 Hz:
    emg,wax,way,waz,wgx,wgy,wgz,eax,eay,eaz,egx,egy,egz\n

  Libraries needed (install via Arduino Library Manager):
    - Adafruit MPU6050
    - Adafruit Unified Sensor
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#define EMG_PIN    34
#define BAUD_RATE  115200
#define LOOP_MS    10
#define SMOOTH_N   8

// Must be defined before any function that uses it
struct IMU6 { float ax, ay, az, gx, gy, gz; };

Adafruit_MPU6050 mpuWrist;  // AD0 -> GND  = 0x68
Adafruit_MPU6050 mpuElbow;  // AD0 -> 3.3V = 0x69

bool wristOK = false;
bool elbowOK = false;

float emgBuf[SMOOTH_N] = {};
int   emgIdx = 0;

// ── EMG ───────────────────────────────────────────────────────────────────────

float readEMG() {
  emgBuf[emgIdx % SMOOTH_N] = (float)analogRead(EMG_PIN);
  emgIdx++;
  float s = 0;
  for (int i = 0; i < SMOOTH_N; i++) s += emgBuf[i];
  return s / SMOOTH_N;
}

// ── IMU ───────────────────────────────────────────────────────────────────────

bool initIMU(Adafruit_MPU6050 &mpu, uint8_t addr, const char *label) {
  if (!mpu.begin(addr)) {
    Serial.printf("# ERR: %s MPU6050 not found at 0x%02X\n", label, addr);
    return false;
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.printf("# OK:  %s MPU6050 at 0x%02X\n", label, addr);
  return true;
}

IMU6 readIMU(Adafruit_MPU6050 &mpu) {
  sensors_event_t a, g, t;
  mpu.getEvent(&a, &g, &t);
  IMU6 out;
  out.ax = a.acceleration.x;
  out.ay = a.acceleration.y;
  out.az = a.acceleration.z;
  out.gx = g.gyro.x;
  out.gy = g.gyro.y;
  out.gz = g.gyro.z;
  return out;
}

IMU6 zeroIMU() {
  IMU6 out;
  out.ax = out.ay = out.az = out.gx = out.gy = out.gz = 0.0f;
  return out;
}

// ── Setup / loop ──────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(BAUD_RATE);
  Wire.begin();
  analogReadResolution(12);

  wristOK = initIMU(mpuWrist, 0x68, "WRIST");
  elbowOK = initIMU(mpuElbow, 0x69, "ELBOW");
}

void loop() {
  float emg   = readEMG();
  IMU6  wrist = wristOK ? readIMU(mpuWrist) : zeroIMU();
  IMU6  elbow = elbowOK ? readIMU(mpuElbow) : zeroIMU();

  Serial.printf(
    "%.1f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f\n",
    emg,
    wrist.ax, wrist.ay, wrist.az, wrist.gx, wrist.gy, wrist.gz,
    elbow.ax, elbow.ay, elbow.az, elbow.gx, elbow.gy, elbow.gz
  );

  delay(LOOP_MS);
}
