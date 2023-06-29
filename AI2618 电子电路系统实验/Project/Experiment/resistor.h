#pragma once
#include "basic.h"
#include "utils.h"

// measure resistor
void measureResistor(int x, int y)
{
  long resistance = 0;
  // large scale
  clearPinMode();
  setPinMode(x, 'H', HIGH);
  setPinMode(y, 'M', LOW);
  delay(1);
  Result[0] = analogRead(TP[x].M);
  if (Result[0] > 1000)
  {
    sprintf(Buffer, "[Resistor]\tInfinity");
    Serial.println(Buffer);
    return;
  }
  resistance = double(510000) / (double(1023) / Result[0] - double(1));
  if (Result[0] <= 20)
  {
    // tiny scale
    clearPinMode();
    setPinMode(x, 'L', HIGH);
    setPinMode(y, 'M', LOW);
    delay(1);
    Result[0] = analogRead(TP[x].M);
    resistance = double(680) / (double(1023) / Result[0] - double(1));
  }
  sprintf(Buffer, "[Resistor]\t%ld ohm", resistance);
  Serial.println(Buffer);
}

// try resistor
bool isResistor(int x, int y)
{
  int last = 0;
  clearPinMode();
  dischargeCapacitor(x, y);
  clearPinMode();
  setPinMode(x, 'L', HIGH);
  setPinMode(y, 'L', LOW);
  delay(200);
  Result[1] = analogRead(TP[x].M);
  Result[0] = analogRead(TP[y].M);
  return Result[0] > 2 || Result[1] < 1015;
}