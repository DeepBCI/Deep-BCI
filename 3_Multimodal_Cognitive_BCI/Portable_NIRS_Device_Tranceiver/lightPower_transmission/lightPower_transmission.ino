
#include <SPI.h>
#include "nRF24L01.h"
#include "RF24.h"
#include "printf.h"

#define DELAY 100

const uint64_t pipe = 0xF0F0F0F0D2LL;                       //pipe setting
RF24 radio(7, 8);                                //pin setting

int trial = 0;
unsigned long time1;
unsigned long time2;

struct pack
  {
    int trialnum;
    int pow1;
    int pow2;  
  };

void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  radio.begin();
  printf_begin();
  if (radio.setDataRate(RF24_2MBPS)) Serial.println("datarate is now 2MBPS");
  radio.setPALevel(RF24_PA_HIGH);
  radio.openWritingPipe(pipe);
  radio.stopListening();
  radio.printDetails();
  delay(1000);
  Serial.println("");
  Serial.println("RF24/examples/GettingStarted");
  Serial.println("");
  
  pinMode(5,OUTPUT);
  pinMode(6,OUTPUT);
  digitalWrite(6,LOW);
  digitalWrite(5,LOW);
  radio.startListening();
  time1 = millis();
  time2 = millis();
}
 
// the loop routine runs over and over again forever:
void loop() {
  radio.stopListening();

  time1 = time2;
  time2 = time1;
  digitalWrite(5,HIGH);

  while(time2 - time1 < 10){
    time2 = millis();
  }
  // read the input on analog pin 0:
  int sensorValue1 = analogRead(A0);
  // print out the value you read:
  digitalWrite(5,LOW);

  digitalWrite(6,HIGH);

  while(time2 - time1 < 20){
    time2 = millis();
  }
  // read the input on analog pin 0:
  int sensorValue2 = analogRead(A0);
  // print out the value you read:
  digitalWrite(6,LOW);

  pack pack1;
  pack1.trialnum = trial;
  pack1.pow1 = sensorValue1; 
  pack1.pow2 = sensorValue2;

  bool ok = radio.write(&pack1, sizeof(pack));

  if (ok){
    Serial.print("Trial: ");
    Serial.println(pack1.trialnum);
    Serial.print("750nm wavelength: ");
    Serial.print(pack1.pow1);
    Serial.println(" was sent");
    Serial.print("840nm wavelength: ");
    Serial.print(pack1.pow2);
    Serial.println(" was sent");    
  }

  else Serial.println("transmission failed");


  radio.startListening();

  
  trial++;
  Serial.println();
  
  // delay in between reads for stability
  while(time2 - time1 < 100){
    time2 = millis();
  }
  
}
