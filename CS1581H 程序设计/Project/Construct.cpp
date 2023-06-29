#pragma once

#include <cstdlib>
#include "Construct.h"

int randInt(int low, int high)
{
	int span = high - low + 1;
	return rand() % span + low;
}

Coordinate::Coordinate(int _x, int _y)
{
	x = _x, y = _y;
}

Coordinate::~Coordinate()
{
	x = y = 0;
}