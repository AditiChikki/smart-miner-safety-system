#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <math.h>

// Wi-Fi credentials
const char* ssid = "aditi";
const char* password = "123456789";

// Remote ML server URL
const String SERVER_URL = "https://97b9-34-75-94-122.ngrok-free.app/predict";

// GPIO Pins
#define MQ137 34   // CO
#define MQ135 32   // CO2
#define MQ7 33     // H2S
#define MQ136 35   // NH3
#define ECG_SENSOR 36
#define BUZZER_PIN 25

// Create web server
WebServer server(80);

// Anomaly status (global)
bool anomalyDetected = false;

// Conversion functions
float convertCO(float voltage) {
  return 1538.46 * pow(voltage, -1.78);
}

float convertCO2(float voltage) {
  return 116.6020682 * pow(voltage, -2.769034857);
}

float convertH2S(float voltage) {
  return pow(10, ((log10(voltage) - 0.72) / -1.3));
}

float convertNH3(float voltage) {
  return pow(10, ((log10(voltage) - 1.47) / -1.67));
}

// ECG Feature Extraction Approximations
float calculatePR(float voltage) {
  return voltage * 120.0;
}

float calculateQT(float voltage) {
  return voltage * 200.0;
}

float calculateST(float voltage) {
  return voltage * 80.0;
}

float calculateHRV(float voltage) {
  return (60000.0 / (voltage * 150.0));
}

// Send sensor data to ML server
bool sendToMLServer(float co, float co2, float h2s, float nh3, float pr, float qt, float st, float hrv) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    DynamicJsonDocument doc(1024);
    doc["data"]["CO"] = co;
    doc["data"]["CO2"] = co2;
    doc["data"]["H2S"] = h2s;
    doc["data"]["NH3"] = nh3;
    doc["data"]["PR"] = pr;
    doc["data"]["QT"] = qt;
    doc["data"]["ST"] = st;
    doc["data"]["HRV"] = hrv;

    String payload;
    serializeJson(doc, payload);

    int httpCode = http.POST(payload);
    if (httpCode > 0) {
      String response = http.getString();
      Serial.println("Server Response: " + response);

      DynamicJsonDocument responseDoc(1024);
      deserializeJson(responseDoc, response);

      if (responseDoc.containsKey("Final_Anomaly")) {
        return responseDoc["Final_Anomaly"];
      }
    } else {
      Serial.println("HTTP Error");
    }
    http.end();
  } else {
    Serial.println("WiFi Disconnected");
  }
  return false; // Default to safe
}

// HTML Page
void handleRoot() {
  // Sensor Readings
  float vCO = (analogRead(MQ137) / 4095.0) * 3.3;
  float vCO2 = (analogRead(MQ135) / 4095.0) * 3.3;
  float vH2S = (analogRead(MQ7) / 4095.0) * 3.3;
  float vNH3 = (analogRead(MQ136) / 4095.0) * 3.3;
  float vECG = (analogRead(ECG_SENSOR) / 4095.0) * 3.3;

  float co =  (convertCO(vCO)/30000)*50;
  float co2 = (convertCO2(vCO2)/40000)*5000;
  float h2s = convertH2S(vH2S);
  float nh3 = convertNH3(vNH3);

  float pr = (calculatePR(vECG)/450)*200;
  float qt = (calculateQT(vECG)/780)*450;
  float st = calculateST(vECG)/2;
  float hrv = (calculateHRV(vECG)/160)*45;

  anomalyDetected = sendToMLServer(co, co2, h2s, nh3, pr, qt, st, hrv);
  digitalWrite(BUZZER_PIN, anomalyDetected ? HIGH : LOW);

  String html = "<html><head><title>ESP32 Sensor Dashboard</title>";
  html += "<style>body{font-family:Arial;text-align:center;background:#f0f0f0;}h1{color:#222;}p{font-size:18px;}strong{color:#444;}</style>";
  html += "<meta http-equiv='refresh' content='5'/>";
  html += "</head><body>";
  html += "<h1>Real-Time Sensor Data</h1>";
  html += "<p><strong>CO:</strong> " + String(co, 2) + " ppm</p>";
  html += "<p><strong>CO2:</strong> " + String(co2, 2) + " ppm</p>";
  html += "<p><strong>H2S:</strong> " + String(h2s, 2) + " ppm</p>";
  html += "<p><strong>NH3:</strong> " + String(nh3, 2) + " ppm</p>";
  html += "<p><strong>PR:</strong> " + String(pr, 2) + " ms</p>";
  html += "<p><strong>QT:</strong> " + String(qt, 2) + " ms</p>";
  html += "<p><strong>ST:</strong> " + String(st, 2) + " ms</p>";
  html += "<p><strong>HRV:</strong> " + String(hrv, 2) + " ms</p>";
  html += "<h2>Status: " + String(anomalyDetected ?  " Normal" : "!!! Anomaly Detected !!!") + "</h2>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi!");
  Serial.print("IP: "); Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.begin();
  Serial.println("Web server started.");
}

void loop() {
  server.handleClient();
}
