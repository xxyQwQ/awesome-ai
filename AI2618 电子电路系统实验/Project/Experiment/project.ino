#include <math.h>
#include "basic.h"
#include "utils.h"
#include "resistor.h"
#include "capacitor.h"
#include "transistor.h"

// initialization
void setup()
{
  Serial.begin(9600);   // start serial
  clearPinMode();       // reset pin mode
}

// main program
void loop()
{
  if (isTransistor())
    measureTransistor();  
  else
  {
    for (int i = 0; i < 3; i++)
      {
        int x = Point[i][0], y = Point[i][1];
        if (isResistor(x, y) && isResistor(y, x))
        {
          measureResistor(x, y);
          break;
        }
        else if (isCapacitor(x, y) && isCapacitor(y, x))
        {
          measureCapacitor(x, y);
          break;
        }
      }
  }
  delay(1000);
}