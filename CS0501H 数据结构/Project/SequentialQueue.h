#pragma once
#include "Exception.h"
#include "Queue.h"

template <typename ElementType>
class SequentialQueue : public Queue<ElementType>
{
private:
	ElementType* elementData;
	int headPosition, tailPosition, totalCapacity;
	void expand();

public:
	SequentialQueue(int size = 10);
	~SequentialQueue();
	bool empty() const;
	void push(const ElementType& element);
	ElementType pop();
	ElementType front() const;
	void clear();
};

template <typename ElementType>
void SequentialQueue<ElementType>::expand()
{
	ElementType* temp = elementData;
	elementData = new ElementType[2 * totalCapacity];
	for (int i = 1; i < totalCapacity; i++)
		elementData[i] = temp[(headPosition + i) % totalCapacity];
	headPosition = 0;
	tailPosition = totalCapacity - 1;
	totalCapacity *= 2;
	delete[] temp;
}

template <typename ElementType>
SequentialQueue<ElementType>::SequentialQueue(int size)
{
	elementData = new ElementType[size];
	totalCapacity = size;
	headPosition = tailPosition = 0;
}

template <typename ElementType>
SequentialQueue<ElementType>::~SequentialQueue()
{
	delete[] elementData;
}

template <typename ElementType>
bool SequentialQueue<ElementType>::empty() const
{
	return headPosition == tailPosition;
}

template <typename ElementType>
void SequentialQueue<ElementType>::push(const ElementType& element)
{
	if ((tailPosition + 1) % totalCapacity == headPosition)
		expand();
	tailPosition = (tailPosition + 1) % totalCapacity;
	elementData[tailPosition] = element;
}

template <typename ElementType>
ElementType SequentialQueue<ElementType>::pop()
{
	if (headPosition == tailPosition)
		throw EmptyContainer("Error: Queue is already empty");
	headPosition = (headPosition + 1) % totalCapacity;
	return elementData[headPosition];
}

template <typename ElementType>
ElementType SequentialQueue<ElementType>::front() const
{
	if (headPosition == tailPosition)
		throw EmptyContainer("Error: Queue is already empty");
	return elementData[(headPosition + 1) % totalCapacity];
}

template <typename ElementType>
void SequentialQueue<ElementType>::clear()
{
	headPosition = tailPosition = 0;
}