#pragma once
#include "basic.h"
#include "utils.h"

// calculate PNP transistor factor
void calculateTransistorPNP(int base, int collector, int emitter)
{
  double beta = 0.0;
  clearPinMode();
  setPinMode(base, 'L', LOW);
  setPinMode(collector, 'M', LOW);
  setPinMode(emitter, 'L', HIGH);
  delay(1);
  Result[0] = analogRead(TP[base].M);
  Result[1] = analogRead(TP[emitter].M);
  if (Result[0] >= 2)
    beta = double(Result[1] - Result[0]) / Result[0];
  else
  {
    clearPinMode();
    setPinMode(base, 'H', LOW);
    setPinMode(collector, 'M', LOW);
    setPinMode(emitter, 'L', HIGH);
    delay(1);
    Result[0] = analogRead(TP[base].M);
    Result[1] = analogRead(TP[emitter].M);
    beta = Result[1] * double(510000) / Result[0] / double(700);
  }
  Serial.print("\tBeta:\t\t");
  Serial.println(beta, 3);
}

// analyse PNP transistor
void analyseTransistorPNP(int x, int y, int z)
{
  sprintf(Buffer, "\tBase:\t\tTP%d", x);
  Serial.println(Buffer);
  // high on TPx, low on TPy and TPz
  clearPinMode();
  setPinMode(x, 'H', HIGH);
  setPinMode(y, 'H', LOW);
  setPinMode(z, 'H', LOW);
  delay(10);
  readTripleResult();
  if (Result[y] >= 100)
  {
    sprintf(Buffer, "\tCollector:\tTP%d", y);
    Serial.println(Buffer);
    sprintf(Buffer, "\tEmitter:\tTP%d", z);
    Serial.println(Buffer);
    calculateTransistorPNP(x, y, z);
  }
  else
  {
    sprintf(Buffer, "\tCollector:\tTP%d", z);
    Serial.println(Buffer);
    sprintf(Buffer, "\tEmitter:\tTP%d", y);
    Serial.println(Buffer);
    calculateTransistorPNP(x, z, y);
  }
}

// calculate NPN transistor factor
void calculateTransistorNPN(int base, int collector, int emitter)
{
  double ib = 0.0, ic = 0.0, beta = 0.0;
  clearPinMode();
  setPinMode(base, 'H', HIGH);
  setPinMode(collector, 'L', HIGH);
  setPinMode(emitter, 'M', LOW);
  delay(10);
  ib = (1023 - analogRead(TP[base].M)) * double(5) / double(1023) / double(510000);
  ic = (1023 - analogRead(TP[collector].M)) * double(5) / double(1023) / double (700);
  beta = ic / ib;
  Serial.print("\tBeta:\t\t");
  Serial.println(beta, 3);
}

// analyse NPN transistor
void analyseTransistorNPN(int x, int y, int z)
{
  sprintf(Buffer, "\tBase:\t\tTP%d", x);
  Serial.println(Buffer);
  // high on TPy, low on TPz, high on TPx with resistor
  clearPinMode();
  setPinMode(x, 'L', HIGH);
  setPinMode(y, 'M', HIGH);
  setPinMode(z, 'M', LOW);
  delay(1);
  Result[y] = analogRead(TP[x].M);
  // high on TP3, low on TP2, high on TP1 with resistor
  clearPinMode();
  setPinMode(x, 'L', HIGH);
  setPinMode(y, 'M', LOW);
  setPinMode(z, 'M', HIGH);
  delay(1);
  Result[z] = analogRead(TP[x].M);
  if (Result[y] > Result[z])
  {
    sprintf(Buffer, "\tCollector:\tTP%d", y);
    Serial.println(Buffer);
    sprintf(Buffer, "\tEmitter:\tTP%d", z);
    Serial.println(Buffer);
    calculateTransistorNPN(x, y, z);
  }
  else
  {
    sprintf(Buffer, "\tCollector:\tTP%d", z);
    Serial.println(Buffer);
    sprintf(Buffer, "\tEmitter:\tTP%d", y);
    Serial.println(Buffer);
    calculateTransistorNPN(x, z, y);
  }
}

// measure transistor
void measureTransistor()
{
  bool feature[4];
  int count = 0;
  for (int i = 1; i <= 3; i++)
  {
    clearPinMode();
    for (int j = 1; j <= 3; j++)
      setPinMode(j, 'L', LOW);
    setPinMode(i, 'L', HIGH);
    delay(1);
    readTripleResult();
    if (feature[i] = isFeature())
      count++;
  }
  if (count == 1)
  {
    Serial.println("[Transistor]\tPNP Type");
    if (feature[1])
      analyseTransistorPNP(1, 2, 3);
    else if (feature[2])
      analyseTransistorPNP(2, 1, 3);
    else
      analyseTransistorPNP(3, 1, 2);
  }
  else if (count == 2)
  {
    Serial.println("[Transistor]\tNPN Type");
    if (!feature[1])
      analyseTransistorNPN(1, 2, 3);
    else if (!feature[2])
      analyseTransistorNPN(2, 1, 3);
    else if (!feature[3])
      analyseTransistorNPN(3, 1, 2);
  }
}

// try transistor
bool isTransistor()
{
  int count = 0;
  for (int i = 1; i <= 3; i++)
  {
    clearPinMode();
    for (int j = 1; j <= 3; j++)
      setPinMode(j, 'L', LOW);
    setPinMode(i, 'L', HIGH);
    delay(1);
    if (analogRead(TP[1].M) >= 100 && analogRead(TP[2].M) >= 100 && analogRead(TP[3].M) >= 100)
      return true;
  }
  return false;
}