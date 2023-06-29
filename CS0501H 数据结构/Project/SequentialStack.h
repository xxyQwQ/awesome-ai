#pragma once
#include <iostream>
#include "Exception.h"
#include "Stack.h"

template <typename ElementType>
class SequentialStack : public Stack<ElementType>
{
private:
	ElementType* elementData;
	int topPosition, totalCapacity;
	void expand();

public:
	SequentialStack(int size = 10);
	~SequentialStack();
	bool empty() const;
	void push(const ElementType& element);
	ElementType pop();
	ElementType top() const;
	void clear();
};

template <typename ElementType>
void SequentialStack<ElementType>::expand()
{
	ElementType* TempData = elementData;
	totalCapacity *= 2;
	elementData = new ElementType[totalCapacity];
	for (int i = 0; i <= topPosition; i++)
		elementData[i] = TempData[i];
	delete[] TempData;
}

template <typename ElementType>
SequentialStack<ElementType>::SequentialStack(int size)
{
	elementData = new ElementType[size];
	totalCapacity = size;
	topPosition = -1;
}

template <typename ElementType>
SequentialStack<ElementType>::~SequentialStack()
{
	delete[] elementData;
}

template <typename ElementType>
bool SequentialStack<ElementType>::empty() const
{
	return topPosition == -1;
}

template <typename ElementType>
void SequentialStack<ElementType>::push(const ElementType& element)
{
	if (topPosition == totalCapacity - 1)
		expand();
	elementData[++topPosition] = element;
}

template <typename ElementType>
ElementType SequentialStack<ElementType>::pop()
{
	if (topPosition == -1)
		throw EmptyContainer("Error: Stack is already empty");
	return elementData[topPosition--];
}

template <typename ElementType>
ElementType SequentialStack<ElementType>::top() const
{
	if (topPosition == -1)
		throw EmptyContainer("Error: Stack is already empty");
	return elementData[topPosition];
}

template <typename ElementType>
void SequentialStack<ElementType>::clear()
{
	topPosition = -1;
}