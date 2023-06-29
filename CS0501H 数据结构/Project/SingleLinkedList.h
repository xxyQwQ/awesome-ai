#pragma once
#include <iostream>
#include "Exception.h"
#include "List.h"

template <typename ElementType>
class SingleLinkedList : public List<ElementType>
{
private:
	struct ListNode
	{
		ElementType data;
		ListNode* next;
		ListNode() : data(), next(nullptr) {}
		ListNode(const ElementType& _data, ListNode* _next = nullptr) : data(_data), next(_next) {}
		~ListNode() {}
	};
	ListNode* head;
	int currentLength;
	ListNode* place(int index) const;

public:
	SingleLinkedList();
	~SingleLinkedList();
	void clear();
	int length() const;
	void insert(int index, const ElementType& element);
	void remove(int index);
	int find(const ElementType& element) const;
	ElementType fetch(int index) const;
	void traverse() const;
	void append(const ElementType& element);
	void erase(int index);
};

template <typename ElementType>
typename SingleLinkedList<ElementType>::ListNode* SingleLinkedList<ElementType>::place(int index) const
{
	ListNode* p = head;
	while (index >= 0)
	{
		index--;
		p = p->next;
	}
	return p;
}

template <typename ElementType>
SingleLinkedList<ElementType>::SingleLinkedList()
{
	head = new ListNode;
	currentLength = 0;
}

template <typename ElementType>
SingleLinkedList<ElementType>::~SingleLinkedList()
{
	clear();
	delete head;
}

template <typename ElementType>
void SingleLinkedList<ElementType>::clear()
{
	ListNode* p = head->next, * q;
	head->next = nullptr;
	while (p != nullptr)
	{
		q = p->next;
		delete p;
		p = q;
	}
	currentLength = 0;
}

template <typename ElementType>
int SingleLinkedList<ElementType>::length() const
{
	return currentLength;
}

template <typename ElementType>
void SingleLinkedList<ElementType>::insert(int index, const ElementType& element)
{
	if (index < 0 || index > currentLength)
		throw IndexExceed();
	ListNode* p = place(index - 1);
	p->next = new ListNode(element, p->next);
	currentLength++;
}

template <typename ElementType>
void SingleLinkedList<ElementType>::remove(int index)
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	ListNode* p = place(index - 1), * q = p->next;
	p->next = q->next;
	delete q;
	currentLength--;
}

template <typename ElementType>
int SingleLinkedList<ElementType>::find(const ElementType& element) const
{
	ListNode* p = head->next;
	int index = 0;
	while (p != nullptr && p->data != element)
	{
		p = p->next;
		index++;
	}
	if (p == nullptr)
		return -1;
	else
		return index;
}

template <typename ElementType>
ElementType SingleLinkedList<ElementType>::fetch(int index) const
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	return place(index)->data;
}

template <typename ElementType>
void SingleLinkedList<ElementType>::traverse() const
{
	ListNode* p = head->next;
	while (p != nullptr)
	{
		std::cout << p->data << "\t";
		p = p->next;
	}
	std::cout << "\n";
}

template <typename ElementType>
void SingleLinkedList<ElementType>::append(const ElementType& element)
{
	insert(currentLength, element);
}

template <typename ElementType>
void SingleLinkedList<ElementType>::erase(int index)
{
	if (index < 0 || index >= currentLength)
		throw IndexExceed();
	ListNode* p = place(index - 1), * q = p->next;
	ElementType target = q->data;
	p->next = q->next;
	delete q;
	int count = 1;
	for (p = head; p->next != nullptr;)
	{
		if (p->next->data == target)
		{
			q = p->next;
			p->next = q->next;
			delete q;
			count++;
		}
		else
			p = p->next;
	}
	currentLength -= count;
}