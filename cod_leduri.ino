#include "BluetoothSerial.h"

#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled! Please run make menuconfig to and enable it
#endif

BluetoothSerial SerialBT;
int received;  // received value will be stored in this variable
char receivedChar;  // received value will be stored as CHAR in this variable

const char turnON1 = 'a';
const char turnOFF1 = 'f';
const int LEDpin1 = 12;

const int LEDpin2 = 14;
const char turnON2 = 'r';
const char turnOFF2 = 'f';

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_A+R");  // Bluetooth device name
  Serial.println("The device started, now you can pair it with Bluetooth!");

  pinMode(LEDpin1, OUTPUT);
  pinMode(LEDpin2, OUTPUT);

  Serial.println("Setup complete");
}

void loop() {
  if (SerialBT.available()) {
    receivedChar = (char)SerialBT.read();

    SerialBT.print("Received: ");  // write on BT app
    SerialBT.println(receivedChar);  // write on BT app      
    Serial.print("Received: ");  // print on serial monitor
    Serial.println(receivedChar);  // print on serial monitor    

    if (receivedChar == turnON1) {
      SerialBT.println("LED1 ON:");  // write on BT app
      Serial.println("LED1 ON:");  // write on serial monitor
      digitalWrite(LEDpin1, HIGH);  // turn the LED ON
    }
    if (receivedChar == turnON2) {
      SerialBT.println("LED2 ON:");  // write on BT app
      Serial.println("LED2 ON:");  // write on serial monitor
      digitalWrite(LEDpin2, HIGH);  // turn the LED ON
    }
    if (receivedChar == turnOFF1) {
      SerialBT.println("LED1 OFF:");  // write on BT app
      Serial.println("LED1 OFF:");  // write on serial monitor
      digitalWrite(LEDpin1, LOW);  // turn the LED off 
    }
    if (receivedChar == turnOFF2) {
      SerialBT.println("LED2 OFF:");  // write on BT app
      Serial.println("LED2 OFF:");  // write on serial monitor
      digitalWrite(LEDpin2, LOW);  // turn the LED off 
    }
  }
  
  delay(20);
}