#pragma once

#include <cstdio>
#include <cmath>
#include <ctime>
#include <string>
#include <graphics.h>
#include <windows.h>
#include "Snake.h"

void Snake::doCreateInterface()
{
	setbkcolor(0x2D2D2D);
	cleardevice();
	setlinecolor(0xFFCC66);
	setfillcolor(0xFFCC66);
	fillrectangle(0, 0, 510, 4);
	fillrectangle(0, 0, 4, 310);
	fillrectangle(506, 0, 510, 310);
	fillrectangle(0, 306, 510, 310);
	fillrectangle(306, 0, 310, 310);
	fillrectangle(310, 190, 510, 195);
	settextstyle(20, 0, _T("Consolas"));
	outtextxy(350, 20, _T("Press Control"));
	outtextxy(340, 50, _T("  [W]: "));
	outtextxy(430, 50, _T("Up"));
	outtextxy(340, 75, _T("  [S]: "));
	outtextxy(430, 75, _T("Down"));
	outtextxy(340, 100, _T("  [A]: "));
	outtextxy(430, 100, _T("Left"));
	outtextxy(340, 125, _T("  [D]: "));
	outtextxy(430, 125, _T("Right"));
	outtextxy(340, 150, _T("[Space]: "));
	outtextxy(430, 150, _T("Pause"));
	doShowScore();
}

Snake::Snake()
{
	srand((int)time(NULL));
	initgraph(510, 310);
	doCreateInterface();
	face = UP;
	score = 0;
	alive = true;
	for (int i = 0; i < 30; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			flag[i][j] = false;
		}
	}
	doGameStart();
}

Snake::~Snake()
{
	body.clear();
	closegraph();
}

void Snake::doDrawBlock(int x, int y)
{
	setlinecolor(0xff0000);
	setfillcolor(0x00ffff);
	fillrectangle(x * 10 + 5, y * 10 + 5, (x + 1) * 10 + 5, (y + 1) * 10 + 5);
}

void Snake::doEraseBlock(int x, int y)
{
	setlinecolor(0x2D2D2D);
	setfillcolor(0x2D2D2D);
	fillrectangle(x * 10 + 5, y * 10 + 5, (x + 1) * 10 + 5, (y + 1) * 10 + 5);
}

void Snake::doShowScore()
{
	setlinecolor(0x2D2D2D);
	setfillcolor(0x2D2D2D);
	fillrectangle(320, 200, 500, 300);
	std::string s = std::to_string(score);
	int l = s.length();
	settextstyle(25, 12, _T("Consolas"));
	outtextxy(345, 220, _T("Game Score"));
	outtextxy((200 - l * 12) / 2 + 310, 255, LPTSTR(s.c_str()));
}

void Snake::doGenerateFood()
{
	int nowX, nowY;
	do
	{
		nowX = randInt(0, 29), nowY = randInt(0, 29);
	} while (flag[nowX][nowY]);
	food = Coordinate(nowX, nowY);
	setlinecolor(0x0000ff);
	setfillcolor(0x0000ff);
	fillcircle(nowX * 10 + 10, nowY * 10 + 10, 4);
}

void Snake::doIncreaseScore()
{
	score += 100;
}

void Snake::doMoveForward()
{
	Coordinate current = body.front();
	int nowX = current.getX(), nowY = current.getY();
	int deltaX = delta[face].getX(), deltaY = delta[face].getY();
	int tempX = nowX + deltaX, tempY = nowY + deltaY;
	if (tempX < 0 || tempY < 0 || tempX >= 30 || tempY >= 30 || flag[tempX][tempY])
	{
		alive = false;
		return;
	}
	else if (tempX == food.getX() && tempY == food.getY())
	{
		doGenerateFood();
		doIncreaseScore();
		doShowScore();
	}
	else
	{
		Coordinate last = body.back();
		int lastX = last.getX(), lastY = last.getY();
		body.pop_back();
		flag[lastX][lastY] = false;
		doEraseBlock(lastX, lastY);
	}
	body.push_front(Coordinate(tempX, tempY));
	flag[tempX][tempY] = true;
	doDrawBlock(tempX, tempY);
}

void Snake::doSetFace(int dirc)
{
	if (((dirc == 0 || dirc == 1) && (face == 2 || face == 3)) || ((dirc == 2 || dirc == 3) && (face == 0 || face == 1)))
	{
		face = dirc;
	}
}

void Snake::doGamePause()
{
	setlinecolor(0x2D2D2D);
	setfillcolor(0x2D2D2D);
	fillrectangle(320, 200, 500, 300);
	settextstyle(25, 12, _T("Consolas"));
	outtextxy(345, 220, _T("Game Paused"));
	settextstyle(15, 0, _T("Consolas"));
	outtextxy(330, 255, _T("Press [Space] to resume"));
}

void Snake::doGameStart()
{
	setlinecolor(0x2D2D2D);
	setfillcolor(0x2D2D2D);
	fillrectangle(320, 200, 500, 300);
	settextstyle(25, 0, _T("Consolas"));
	outtextxy(380, 235, _T("Ready"));
	Sleep(800);
	fillrectangle(320, 200, 500, 300);
	for (int i = 3; i >= 1; i--)
	{
		std::string s = std::to_string(i);
		outtextxy(400, 235, LPTSTR(s.c_str()));
		Sleep(800);
	}
	doShowScore();
	for (int i = 1; i <= 3; i++)
	{
		int tempX = randInt(10, 20), tempY = randInt(10, 20);
		body.push_front(Coordinate(tempX, tempY));
		flag[tempX][tempY] = true;
	}
	doGenerateFood();
}

void Snake::doGameOver()
{
	setlinecolor(0x2D2D2D);
	setfillcolor(0x2D2D2D);
	fillrectangle(320, 200, 500, 300);
	settextstyle(25, 0, _T("Consolas"));
	for (int i = 1; i <= 3; i++)
	{
		outtextxy(355, 235, _T("Game Over"));
		Sleep(500);
		fillrectangle(320, 200, 500, 300);
		Sleep(300);
	}
	std::string s = std::to_string(score);
	int l = s.length();
	settextstyle(25, 12, _T("Consolas"));
	outtextxy(340, 220, _T("Final Score"));
	outtextxy((200 - l * 12) / 2 + 310, 255, LPTSTR(s.c_str()));
	Sleep(1000);
	while (!body.empty())
	{
		Coordinate last = body.back();
		int lastX = last.getX(), lastY = last.getY();
		setlinecolor(0x2D2D2D);
		setfillcolor(0x2D2D2D);
		fillrectangle(lastX * 10 + 5, lastY * 10 + 5, (lastX + 1) * 10 + 5, (lastY + 1) * 10 + 5);
		body.pop_back();
		Sleep(200);
	}
}

void Snake::doShowExit()
{
	fillrectangle(320, 200, 500, 300);
	settextstyle(15, 0, _T("Consolas"));
	outtextxy(335, 240, _T("Press any key to exit"));
}