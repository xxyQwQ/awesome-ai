#pragma once

int randInt(int low = 0, int high = 32767);

class Coordinate
{
private:
	int x, y;
public:
	Coordinate(int _x = 0, int _y = 0);
	~Coordinate();
	const int getX() const
	{
		return x;
	}
	const int getY() const
	{
		return y;
	}
};