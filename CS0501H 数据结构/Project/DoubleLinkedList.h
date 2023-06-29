#pragma once
#include <iostream>
#include "List.h"

template <typename ElementType>
class DoubleLinkedList : public List<ElementType>
{
private:
	struct ListNode
	{
		ElementType data;
		ListNode* prev, * next;
		ListNode() : data(), prev(nullptr), next(nullptr) {}
		ListNode(const ElementType& _data, ListNode* _prev = nullptr, ListNode* _next = nullptr) : data(_data), prev(_prev), next(_next) {}
		~ListNode() {}
	};
	ListNode* head, * tail;
	int currentLength;
	ListNode* place(int index) const;

public:
	DoubleLinkedList();
	~DoubleLinkedList();
	void clear();
	int length() const;
	void insert(int index, const ElementType& element);
	void remove(int index);
	int find(const ElementType& element) const;
	ElementType fetch(int index) const;
	void traverse() const;
	void push_front(const ElementType& element);
	void push_back(const ElementType& element);
	void reverse();
};

template <typename ElementType>
typename DoubleLinkedList<ElementType>::ListNode* DoubleLinkedList<ElementType>::place(int index) const
{
	ListNode* p = head;
	while (index >= 0)
	{
		p = p->next;
		index--;
	}
	return p;
}

template <typename ElementType>
DoubleLinkedList<ElementType>::DoubleLinkedList()
{
	head = new ListNode;
	head->next = tail = new ListNode;
	tail->prev = head;
	currentLength = 0;
}

template <typename ElementType>
DoubleLinkedList<ElementType>::~DoubleLinkedList()
{
	clear();
	delete head;
	delete tail;
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::clear()
{
	ListNode* p = head->next, * q;
	head->next = tail;
	tail->prev = head;
	while (p != tail)
	{
		q = p->next;
		delete p;
		p = q;
	}
	currentLength = 0;
}

template <typename ElementType>
int DoubleLinkedList<ElementType>::length() const
{
	return currentLength;
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::insert(int index, const ElementType& element)
{
	if (index < 0 || index > currentLength)
		throw IndexExceed();
	ListNode* p = place(index), * q;
	q = new ListNode(element, p->prev, p);
	p->prev->next = q;
	p->prev = q;
	currentLength++;
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::remove(int index)
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	ListNode* p = place(index);
	p->prev->next = p->next;
	p->next->prev = p->prev;
	delete p;
	currentLength--;
}

template <typename ElementType>
int DoubleLinkedList<ElementType>::find(const ElementType& element) const
{
	ListNode* p = head->next;
	int index = 0;
	while (p != tail && p->data != element)
	{
		p = p->next;
		index++;
	}
	if (p == tail)
		return -1;
	else
		return index;
}

template <typename ElementType>
ElementType DoubleLinkedList<ElementType>::fetch(int index) const
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	return place(index)->data;
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::traverse() const
{
	ListNode* p = head->next;
	while (p != tail)
	{
		std::cout << p->data << "\t";
		p = p->next;
	}
	std::cout << "\n";
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::push_front(const ElementType& element)
{
	insert(0, element);
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::push_back(const ElementType& element)
{
	insert(currentLength, element);
}

template <typename ElementType>
void DoubleLinkedList<ElementType>::reverse()
{
	ListNode* p = head->next, * q;
	head->prev = head->next;
	head->next = nullptr;
	while (p != tail)
	{
		q = p->prev;
		p->prev = p->next;
		p->next = q;
		p = p->prev;
	}
	tail->next = tail->prev;
	tail->prev = nullptr;
	q = head;
	head = tail;
	tail = q;
}