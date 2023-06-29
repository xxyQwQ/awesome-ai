#pragma once
#include <iostream>
#include "Exception.h"
#include "List.h"

template <typename ElementType>
class SequentialList : public List<ElementType>
{
private:
	ElementType* elementData;
	int currentLength, totalCapacity;
	void expand();

public:
	SequentialList(int size = 10);
	~SequentialList();
	void clear();
	int length() const;
	int capacity() const;
	void insert(int index, const ElementType& element);
	void remove(int index);
	int find(const ElementType& element) const;
	ElementType fetch(int index) const;
	void traverse() const;
	ElementType back();
	void append(const ElementType& element);
	void cancel();
	void resize(int size);
	ElementType& operator[](int index);
	ElementType& operator[](int index) const;
};

template <typename ElementType>
void SequentialList<ElementType>::expand()
{
	ElementType* tempData = elementData;
	totalCapacity *= 2;
	elementData = new ElementType[totalCapacity];
	for (int i = 0; i < currentLength; i++)
		elementData[i] = tempData[i];
	delete[] tempData;
}

template <typename ElementType>
SequentialList<ElementType>::SequentialList(int size)
{
	currentLength = 0;
	totalCapacity = size;
	elementData = new ElementType[size];
}

template <typename ElementType>
SequentialList<ElementType>::~SequentialList()
{
	delete[] elementData;
}

template <typename ElementType>
void SequentialList<ElementType>::clear()
{
	currentLength = 0;
}

template <typename ElementType>
int SequentialList<ElementType>::length() const
{
	return currentLength;
}

template <typename ElementType>
int SequentialList<ElementType>::capacity() const
{
	return totalCapacity;
}

template <typename ElementType>
void SequentialList<ElementType>::insert(int index, const ElementType& element)
{
	if (index < 0 || index > currentLength)
		throw IndexExceed();
	if (currentLength == totalCapacity)
		expand();
	for (int i = currentLength; i > index; i--)
		elementData[i] = elementData[i - 1];
	elementData[index] = element;
	currentLength++;
}

template <typename ElementType>
void SequentialList<ElementType>::remove(int index)
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	for (int i = index; i < currentLength - 1; i++)
		elementData[i] = elementData[i + 1];
	currentLength--;
}

template <typename ElementType>
int SequentialList<ElementType>::find(const ElementType& element) const
{
	int i;
	for (i = 0; i < currentLength && elementData[i] != element; i++)
		;
	if (i == currentLength)
		return -1;
	else
		return i;
}

template <typename ElementType>
ElementType SequentialList<ElementType>::fetch(int index) const
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	return elementData[index];
}

template <typename ElementType>
void SequentialList<ElementType>::traverse() const
{
	for (int i = 0; i < currentLength; i++)
		std::cout << elementData[i] << "\t";
	std::cout << "\n";
}

template <typename ElementType>
ElementType SequentialList<ElementType>::back()
{
	return elementData[currentLength - 1];
}

template <typename ElementType>
void SequentialList<ElementType>::append(const ElementType& element)
{
	insert(currentLength, element);
}

template <typename ElementType>
void SequentialList<ElementType>::cancel()
{
	remove(currentLength - 1);
}

template <typename ElementType>
void SequentialList<ElementType>::resize(int size)
{
	if (size <= totalCapacity)
		throw InvalidModify();
	ElementType* tempData = elementData;
	totalCapacity = size;
	elementData = new ElementType[totalCapacity];
	for (int i = 0; i < currentLength; i++)
		elementData[i] = tempData[i];
	delete[] tempData;
}

template <typename ElementType>
ElementType& SequentialList<ElementType>::operator[](int index)
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	return elementData[index];
}

template <typename ElementType>
ElementType& SequentialList<ElementType>::operator[](int index) const
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	return elementData[index];
}