#include <ESP8266WiFi.h>  // <--- Changed from WiFi.h (ESP32) to ESP8266WiFi.h (NodeMCU)
#include <WiFiUdp.h>

// --- CONFIGURATION ---
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
const int localPort = 4210;

// --- PIN DEFINITIONS (NodeMCU) ---
const int motorPin1 = D1; 
const int motorPin2 = D2;
const int soilSensorPin = A0;

// --- SETTINGS ---
// Adjust this threshold! (0 = Wet, 1024 = Dry)
// Usually ~700 is a good starting point for "Dry".
const int DRY_THRESHOLD = 700; 

// How long to stay in Manual Mode before going back to Auto (in milliseconds)
const unsigned long MANUAL_TIMEOUT = 60000; // 60 seconds

// --- VARIABLES ---
WiFiUDP udp;
char packetBuffer[255];

enum SystemMode { MODE_AUTO, MODE_MANUAL_RUN, MODE_MANUAL_STOP };
SystemMode currentMode = MODE_AUTO;
unsigned long lastGestureTime = 0;

void setup() {
  Serial.begin(115200);

  // Set pins
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  pinMode(soilSensorPin, INPUT);

  // Default OFF
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);

  // Connect to Wi-Fi
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected.");
  Serial.print("NodeMCU IP Address: ");
  Serial.println(WiFi.localIP()); // <--- Put this IP in your Python code

  // Start UDP
  udp.begin(localPort);
  Serial.printf("Listening on port %d\n", localPort);
}

void loop() {
  // 1. Check for incoming gestures
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(packetBuffer, 255);
    if (len > 0) packetBuffer[len] = 0;
    
    String command = String(packetBuffer);
    Serial.print("Gesture Received: ");
    Serial.println(command);

    // Update Mode based on Gesture
    if (command == "RUN") {
      currentMode = MODE_MANUAL_RUN;
      lastGestureTime = millis(); // Reset timeout timer
    } 
    else if (command == "STOP") {
      currentMode = MODE_MANUAL_STOP;
      lastGestureTime = millis(); // Reset timeout timer
    }
  }

  // 2. Check Timeout (Revert to Auto if no gestures for 60 seconds)
  if (currentMode != MODE_AUTO && (millis() - lastGestureTime > MANUAL_TIMEOUT)) {
    Serial.println("Timeout! Reverting to AUTO Mode.");
    currentMode = MODE_AUTO;
  }

  // 3. Control Motor based on Mode
  switch (currentMode) {
    
    case MODE_MANUAL_RUN:
      // Force Motor ON
      digitalWrite(motorPin1, HIGH);
      digitalWrite(motorPin2, LOW);
      break;

    case MODE_MANUAL_STOP:
      // Force Motor OFF
      digitalWrite(motorPin1, LOW);
      digitalWrite(motorPin2, LOW);
      break;

    case MODE_AUTO:
      // Read Soil Sensor
      int moistureValue = analogRead(soilSensorPin);
      
      // Debug print (helps you calibrate threshold)
      // Serial.print("Soil Moisture: ");
      // Serial.println(moistureValue);

      if (moistureValue > DRY_THRESHOLD) {
        // Soil is DRY -> Turn ON
        digitalWrite(motorPin1, HIGH);
        digitalWrite(motorPin2, LOW);
      } else {
        // Soil is WET -> Turn OFF
        digitalWrite(motorPin1, LOW);
        digitalWrite(motorPin2, LOW);
      }
      break;
  }
  
  delay(100); // Small delay for stability
}