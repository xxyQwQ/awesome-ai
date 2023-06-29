#pragma once
#include "Exception.h"
#include "Compare.h"
#include "Queue.h"

template <typename ElementType, typename CompareRule = Less<ElementType>>
class PriorityQueue : public Queue<ElementType>
{
private:
	ElementType* elementData;
	int currentLength, totalCapacity;
	CompareRule compare;
	void expand();
	void update(int hole);
	void construct();

public:
	PriorityQueue(int size = 100);
	PriorityQueue(const ElementType content[], int size);
	~PriorityQueue();
	bool empty() const;
	void push(const ElementType& element);
	ElementType pop();
	ElementType front() const;
	ElementType top() const;
	void clear();
};

template <typename ElementType, typename CompareRule>
void PriorityQueue<ElementType, CompareRule>::expand()
{
	ElementType* tempData = elementData;
	totalCapacity <<= 1;
	elementData = new ElementType[totalCapacity + 1];
	for (int i = 0; i <= currentLength; i++)
		elementData[i] = tempData[i];
	delete[] tempData;
}

template <typename ElementType, typename CompareRule>
void PriorityQueue<ElementType, CompareRule>::update(int hole)
{
	ElementType temp = elementData[hole];
	for (int next = 0; (hole << 1) <= currentLength; hole = next)
	{
		next = hole << 1;
		if (next != currentLength && compare(elementData[next | 1], elementData[next]))
			next |= 1;
		if (!compare(elementData[next], temp))
			break;
		elementData[hole] = elementData[next];
	}
	elementData[hole] = temp;
}

template <typename ElementType, typename CompareRule>
void PriorityQueue<ElementType, CompareRule>::construct()
{
	for (int i = currentLength >> 1; i; i--)
		update(i);
}

template <typename ElementType, typename CompareRule>
PriorityQueue<ElementType, CompareRule>::PriorityQueue(int size) : currentLength(0), totalCapacity(size)
{
	elementData = new ElementType[size];
}

template <typename ElementType, typename CompareRule>
PriorityQueue<ElementType, CompareRule>::PriorityQueue(const ElementType content[], int size) : currentLength(size), totalCapacity(size + 10)
{
	elementData = new ElementType[totalCapacity];
	for (int i = 0; i < size; i++)
		elementData[i + 1] = content[i];
	construct();
}

template <typename ElementType, typename CompareRule>
PriorityQueue<ElementType, CompareRule>::~PriorityQueue()
{
	delete[] elementData;
}

template <typename ElementType, typename CompareRule>
bool PriorityQueue<ElementType, CompareRule>::empty() const
{
	return currentLength == 0;
}

template <typename ElementType, typename CompareRule>
void PriorityQueue<ElementType, CompareRule>::push(const ElementType& element)
{
	if (currentLength == totalCapacity - 1)
		expand();
	int hole = ++currentLength;
	for (; hole > 1 && compare(element, elementData[hole >> 1]); hole >>= 1)
		elementData[hole] = elementData[hole >> 1];
	elementData[hole] = element;
}

template <typename ElementType, typename CompareRule>
ElementType PriorityQueue<ElementType, CompareRule>::pop()
{
	if (!currentLength)
		throw EmptyContainer("Error: Heap is already empty");
	ElementType result = elementData[1];
	elementData[1] = elementData[currentLength--];
	update(1);
	return result;
}

template <typename ElementType, typename CompareRule>
ElementType PriorityQueue<ElementType, CompareRule>::front() const
{
	return elementData[1];
}

template <typename ElementType, typename CompareRule>
ElementType PriorityQueue<ElementType, CompareRule>::top() const
{
	return elementData[1];
}

template <typename ElementType, typename CompareRule>
void PriorityQueue<ElementType, CompareRule>::clear()
{
	currentLength = 0;
}