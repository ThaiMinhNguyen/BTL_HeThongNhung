// Arduino code for drowning detection alert system with active buzzer
// Compatible with Arduino UNO R3

const int buzzerPin = 9;  // Connect active buzzer to digital pin 9
const int ledPin = 13;    // Built-in LED for visual indication

boolean drowningDetected = false;
String inputString = "";
boolean stringComplete = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize buzzer and LED pins as outputs
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  
  // Initially turn off the buzzer and LED
  digitalWrite(buzzerPin, LOW);
  digitalWrite(ledPin, LOW);
  
  // Reserved for string input
  inputString.reserve(200);
}

void loop() {
  // Process incoming commands when a complete string is received
  if (stringComplete) {
    if (inputString.indexOf("DROWNING_ALERT") >= 0) {
      // Drowning alert received - activate alarm
      drowningDetected = true;
      Serial.println("ALERT_ACTIVATED");
    } 
    else if (inputString.indexOf("STOP_ALERT") >= 0) {
      // Stop alert command received
      drowningDetected = false;
      Serial.println("ALERT_STOPPED");
    }
    
    // Clear the string for new input
    inputString = "";
    stringComplete = false;
  }
  
  // Activate buzzer and LED when drowning is detected
  if (drowningDetected) {
    // Pattern: Rapid beeping for urgency
    digitalWrite(buzzerPin, HIGH);
    digitalWrite(ledPin, HIGH);
    delay(200);
    digitalWrite(buzzerPin, LOW);
    digitalWrite(ledPin, LOW);
    delay(100);
  } else {
    // No drowning detected, keep buzzer and LED off
    digitalWrite(buzzerPin, LOW);
    digitalWrite(ledPin, LOW);
  }
}

// SerialEvent occurs whenever new data comes in the serial port
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    // Add character to input string
    inputString += inChar;
    
    // If the incoming character is a newline, set a flag so the main loop can
    // do something about it
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
} 