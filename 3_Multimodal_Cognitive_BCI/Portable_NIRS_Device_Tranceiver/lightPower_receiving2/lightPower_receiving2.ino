
#include <SPI.h>
#include "nRF24L01.h"
#include "RF24.h"
#include "printf.h"

/*
#define CE_PIN 7
#define CSN_PIN 8
*/

const uint64_t pipe = 0xF0F0F0F0D2LL;                       //pipe setting
RF24 radio(7, 8);                                //pin setting

struct pack
  {
    int trialnum;
    int pow1;
    int pow2;  
  };

void setup() {

  Serial.begin(9600);
  printf_begin();
  radio.begin();
  radio.startListening();
  radio.stopListening();
  radio.setDataRate(RF24_2MBPS);
  radio.setPALevel(RF24_PA_HIGH);
  radio.openReadingPipe(1,pipe);
  //if(radio.setDataRate(RF24_2MBPS)) Serial.println("datarate is now 250kbps");
  //radio.setRetries(15,15);
  //radio.printDetails();
  radio.startListening();
  delay(1000);
  //Serial.println("Receiving signals in sequences");
 
}

void loop() {

    pack pack1;
    
    if(radio.available()){
   
    bool done = false;

    while(!done){
    done = radio.read( &pack1, sizeof(pack));}
    

    
    byte d[12] = {0,};
    inttobyte(pack1.trialnum, pack1.pow1, pack1.pow2, d);
  

  String str1 = String(pack1.trialnum);
  String str2 = String(pack1.pow1);
  String str3 = String(pack1.pow2);

  char evoke = Serial.read();

  //if(evoke =! 'a'){

  Serial.print("<");
  Serial.print(str1);
  Serial.print(",");
  Serial.print(str2);
  Serial.print(",");
  Serial.print(str3);
  Serial.println(">");
  
    }
}


byte* inttobyte(int a, int b, int c, byte* buf) {

   buf[0] = a & 0xFF;
   buf[1] = (a >> 8) & 0xFF;
   buf[2] = (a >> 16) & 0xFF;
   buf[3] = (a >> 24) & 0xFF;
   buf[4] = (a >> 32) & 0xFF;
   buf[5] = (a >> 40) & 0xFF;
   buf[6] = (a >> 48) & 0xFF;
   buf[7] = (a >> 56) & 0xFF;
   buf[8] = (a >> 64) & 0xFF;
   buf[9] = (a >> 72) & 0xFF;
   buf[10] = (a >> 80) & 0xFF;
   buf[11] = (a >> 88) & 0xFF;

   return buf;

}

int bytetoint(byte* buf){

  int value = (int)((buf[3] & 0xFF) << 24) | ((buf[2] & 0xFF) << 16) | ((buf[1] & 0xFF) << 8) | (buf[0] & 0xFF);

  return value;
  
}

byte* concatbyte(byte* a, byte* b, byte* c){

  byte buf[12];

    for(int i = 0; i < 4; i++){
      buf[i] = a[i];
    }

    for(int i = 0; i < 4; i++){
      buf[i + 4] = b[i];
    }
    
    for(int i = 0; i < 4; i++){
      buf[i + 8] = c[i];
    }
    
}


