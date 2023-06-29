#pragma once
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "Exception.h"
#include "LinkedQueue.h"
#include "PriorityQueue.h"

class SimulatorQueue
{
private:
	struct Event
	{
		int type, time; // type: 0 for arrival, 1 for leave
		Event(int _type = 0, int _time = 0) :type(_type), time(_time) {}
		bool operator<(const Event& another) const
		{
			return time < another.time;
		}
	};
	int ServerCount, CustomerCount;
	int IntervalLow, IntervalHigh;
	int ServiceLow, ServiceHigh;
public:
	SimulatorQueue();
	~SimulatorQueue() {}
	int GetAverageTime();
};

SimulatorQueue::SimulatorQueue()
{
	std::cout << "Input lower and upper bounds of interval time: ";
	std::cin >> IntervalLow >> IntervalHigh;
	std::cout << "Input lower and upper bounds of service time: ";
	std::cin >> ServiceLow >> ServiceHigh;
	std::cout << "Input number of servers: ";
	std::cin >> ServerCount;
	std::cout << "Input number of customers: ";
	std::cin >> CustomerCount;
	srand(int(time(NULL)));
}

int SimulatorQueue::GetAverageTime()
{
	int BusyCount = 0, TotalTime = 0, CurrentTime;
	Event CurrentEvent(0, 0);
	LinkedQueue<Event> CustomerQueue;
	PriorityQueue<Event> EventQueue;
	for (int i = 0; i < CustomerCount; i++)
	{
		int interval = IntervalLow + (rand() % (IntervalHigh - IntervalLow + 1));
		CurrentEvent.time += interval;
		EventQueue.push(CurrentEvent);
	}
	while (!EventQueue.empty())
	{
		CurrentEvent = EventQueue.pop();
		CurrentTime = CurrentEvent.time;
		if (CurrentEvent.type == 0)
		{
			if (BusyCount == ServerCount)
				CustomerQueue.push(CurrentEvent);
			else
			{
				BusyCount++;
				int service = ServiceLow + (rand() % (ServiceHigh - ServiceLow + 1));
				CurrentEvent.time += service;
				CurrentEvent.type = 1;
				EventQueue.push(CurrentEvent);
			}
		}
		else if (CurrentEvent.type == 1)
		{
			if (CustomerQueue.empty())
				BusyCount--;
			else
			{
				CurrentEvent = CustomerQueue.pop();
				TotalTime += CurrentTime - CurrentEvent.time;
				int service = ServiceLow + (rand() % (ServiceHigh - ServiceLow + 1));
				CurrentEvent.time = CurrentTime + service;
				CurrentEvent.type = 1;
				EventQueue.push(CurrentEvent);
			}
		}
	}
	return TotalTime / CustomerCount;
}