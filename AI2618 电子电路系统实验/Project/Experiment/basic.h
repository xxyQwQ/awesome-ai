#pragma once

// short for unsigned type
typedef unsigned long ulong;

// define test pin type
struct TestPin
{
  int M, L, H;
  TestPin(int _M = 0, int _L = 0, int _H = 0): M(_M), L(_L), H(_H) {}
};

// test pins
const TestPin TP[4] = {
  TestPin(0, 0, 0),
  TestPin(A0, 8, 9),
  TestPin(A1, 10, 11),
  TestPin(A2, 12, 13),
};

// connection information
const int Point[6][2] = {
  {1, 2},
  {1, 3},
  {2, 3},
};

// measure result
int Result[4];

// output buffer
char Buffer[100];