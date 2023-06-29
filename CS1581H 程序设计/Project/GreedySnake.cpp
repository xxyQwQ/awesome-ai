#include <conio.h>
#include <windows.h>
#include "Snake.h"

int main()
{
    Snake myGame;
    while (_kbhit())
    {
        char c = _getch();
    }
    while (true)
    {
        Sleep(200);
        while (_kbhit())
        {
            char c = _getch();
            if (c == 'w')
            {
                myGame.doSetFace(0);
            }
            else if (c == 's')
            {
                myGame.doSetFace(1);
            }
            else if (c == 'a')
            {
                myGame.doSetFace(2);
            }
            else if (c == 'd')
            {
                myGame.doSetFace(3);
            }
            else if (c == ' ')
            {
                myGame.doGamePause();
                while (!_kbhit())
                {
                    c = _getch();
                    if (c == ' ')
                    {
                        break;
                    }
                }
                myGame.doShowScore();
            }
            break;
        }
        myGame.doMoveForward();
        if (!myGame.isStillAlive())
        {
            myGame.doGameOver();
            while (_kbhit())
            {
                char c = _getch();
            }
            myGame.doShowExit();
            while (!_kbhit())
            {
                char c = _getch();
                break;
            }
            break;
        }
    }
    return 0;
}