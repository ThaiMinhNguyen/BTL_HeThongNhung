// Arduino code for drowning detection alert system with 3-pin buzzer module
// Compatible with Arduino UNO R3

// ====== CONNECTION GUIDE ======
// Buzzer Module Connections:
// VCC pin  --> Connect to 5V on Arduino
// I/O pin  --> Connect to Digital Pin 9 on Arduino
// GND pin  --> Connect to GND on Arduino
// ============================== 

const int buzzerPin = 7;  // Connect buzzer I/O pin to digital pin 9
const int ledPin = 13;    // Built-in LED for visual indication

boolean drowningDetected = false;
String inputString = "";
boolean stringComplete = false;

// Define logic levels for the buzzer
// For LOW LEVEL TRIGGER buzzer (buzzer sounds when I/O pin is LOW)
const int BUZZER_ON = LOW;   // LOW activates the buzzer (makes sound)
const int BUZZER_OFF = HIGH; // HIGH deactivates the buzzer (silent)

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize buzzer I/O pin and LED pin as outputs
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  
  // IMPORTANT: Make sure buzzer is OFF at startup
  digitalWrite(buzzerPin, BUZZER_OFF);  // Set HIGH to keep buzzer OFF
  digitalWrite(ledPin, LOW);
  
  // Double-check buzzer is off by setting it again after a short delay
  delay(100);
  digitalWrite(buzzerPin, BUZZER_OFF);  // Confirm buzzer is OFF
  
  // Send a confirmation message
  Serial.println("Drowning detection system started with LOW LEVEL TRIGGER buzzer. Buzzer is OFF.");
  
  // Reserved for string input
  inputString.reserve(200);
  
  // Double confirmation that buzzer is OFF
  delay(500);
  digitalWrite(buzzerPin, BUZZER_OFF);  // Final confirmation buzzer is OFF
}

void loop() {
  // Process incoming commands when a complete string is received
  if (stringComplete) {
    if (inputString.indexOf("DROWNING_ALERT") >= 0) {
      // Drowning alert received - activate alarm
      drowningDetected = true;
      digitalWrite(buzzerPin, BUZZER_OFF);  // Reset buzzer state before pattern starts
      Serial.println("ALERT_ACTIVATED");
    } 
    else if (inputString.indexOf("STOP_ALERT") >= 0) {
      // Stop alert command received
      drowningDetected = false;
      digitalWrite(buzzerPin, BUZZER_OFF);  // Ensure buzzer is OFF
      digitalWrite(ledPin, LOW);            // Ensure LED is OFF
      Serial.println("ALERT_STOPPED");
    }
    
    // Clear the string for new input
    inputString = "";
    stringComplete = false;
  }
  
  // Activate buzzer and LED when drowning is detected
  if (drowningDetected) {
    // Pattern: Rapid beeping for urgency
    digitalWrite(buzzerPin, BUZZER_ON);  // Turn ON buzzer (LOW for active-low buzzer)
    digitalWrite(ledPin, HIGH);          // Turn ON LED
    delay(200);
    digitalWrite(buzzerPin, BUZZER_OFF); // Turn OFF buzzer (HIGH for active-low buzzer)
    digitalWrite(ledPin, LOW);           // Turn OFF LED
    delay(100);
  } else {
    // No drowning detected, ensure buzzer and LED are off
    digitalWrite(buzzerPin, BUZZER_OFF); // Ensure buzzer is OFF (HIGH for active-low buzzer)
    digitalWrite(ledPin, LOW);           // Ensure LED is OFF
    delay(50);                           // Short delay to prevent CPU hogging
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
