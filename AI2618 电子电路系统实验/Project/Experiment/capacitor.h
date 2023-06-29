#pragma once
#include "basic.h"
#include "utils.h"

// calculate capacitance
double calculateCapacitance(int x, int y)
{
  ulong start = 0, finish = 0;
  long last = 0, count = 0;
  int value = 0;
  clearPinMode();
  dischargeCapacitor(x, y);
  clearPinMode();
  setPinMode(x, 'L', HIGH);
  setPinMode(y, 'M', LOW);
  count = 0;
  value = analogRead(TP[x].M);
  start = micros();
  while (++count <= 500 && value < 647)
    value = analogRead(TP[x].M);
  finish = micros();
  if (count > 500)
    return double(0);
  last = max(long(finish - start), 0);
  if (last > 500)
    return last / double(720);
  clearPinMode();
  dischargeCapacitor(x, y);
  clearPinMode();
  setPrescaler('L');
  setPinMode(x, 'H', HIGH);
  setPinMode(y, 'M', LOW);
  count = 0;
  value = analogRead(TP[x].M);
  start = micros();
  while (++count <= 5000 && value < 647)
    value = analogRead(TP[x].M);
  finish = micros();
  setPrescaler('H');
  if (count > 5000)
    return double(0);
  last = max(long(finish - start) - 20, 0);
  return last / double(510000);
}

// measure capacitor
void measureCapacitor(int x, int y)
{
  double capacitance = calculateCapacitance(x, y);
  Serial.print("[Capacitor]\t");
  Serial.print(capacitance, 6);
  Serial.println(" uF");
}

// try capacitor
bool isCapacitor(int x, int y)
{
  double capacitance = calculateCapacitance(x, y);
  return capacitance > 1e-5;
}