#pragma once

#include <list>
#include "Construct.h"

class Snake
{
private:
	enum Direction { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
	Coordinate delta[5] = { Coordinate(0,-1),Coordinate(0,1),Coordinate(-1,0),Coordinate(1,0) };
	int face, score;
	bool flag[30][30], alive;
	std::list<Coordinate> body;
	Coordinate food;
	void doDrawBlock(int x, int y);
	void doEraseBlock(int x, int y);
	void doCreateInterface();
	void doGenerateFood();
	void doIncreaseScore();
public:
	Snake();
	~Snake();
	void doMoveForward();
	void doSetFace(int dirc);
	void doShowScore();
	void doGamePause();
	void doGameStart();
	void doGameOver();
	void doShowExit();
	bool isStillAlive()
	{
		return alive;
	}
};