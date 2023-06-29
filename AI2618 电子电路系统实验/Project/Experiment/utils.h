#pragma once
#include "basic.h"

// set prescaler
void setPrescaler(char type)
{
  ADCSRA &= ~(bit (ADPS0) | bit (ADPS1) | bit (ADPS2));
  if (type == 'H')      // 128
    ADCSRA |= bit (ADPS0) | bit (ADPS1) | bit (ADPS2);
  else if (type == 'L') // 16
    ADCSRA |= bit (ADPS2);
}

// reset pin mode
void clearPinMode()
{
  for (int i = 1; i <= 3; i++)
  {
    pinMode(TP[i].M, INPUT);
    pinMode(TP[i].L, INPUT);
    pinMode(TP[i].H, INPUT);
  }
  delay(1);
}

// set pin mode
void setPinMode(int x, char t, int v)
{
  if (t == 'M')
  {
    pinMode(TP[x].M, OUTPUT);
    digitalWrite(TP[x].M, v);
  }
  else if (t == 'L')
  {
    pinMode(TP[x].L, OUTPUT);
    digitalWrite(TP[x].L, v);
  }
  else if (t == 'H')
  {
    pinMode(TP[x].H, OUTPUT);
    digitalWrite(TP[x].H, v);
  }
}

// discharge capacitor
void dischargeCapacitor(int x, int y)
{
  clearPinMode();
  setPinMode(x, 'L', LOW);
  setPinMode(y, 'M', LOW);
  Result[0] = analogRead(TP[x].M);
  while (Result[0] > 1)
    Result[0] = analogRead(TP[x].M);
  delay(10);
}

// transistor feature
bool isFeature()
{
  if (Result[1] >= 1000 && Result[2] <= 10 && Result[3] <= 10)
    return true;
  if (Result[1] <= 10 && Result[2] >= 1000 && Result[3] <= 10)
    return true;
  if (Result[1] <= 10 && Result[2] <= 10 && Result[3] >= 1000)
    return true;
  return false;
}

// read three fixed results
void readTripleResult()
{
  Result[1] = analogRead(TP[1].M);
  Result[2] = analogRead(TP[2].M);
  Result[3] = analogRead(TP[3].M);
}