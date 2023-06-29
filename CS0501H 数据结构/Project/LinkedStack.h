#pragma once
#include <iostream>
#include "Exception.h"
#include "Stack.h"

template <typename ElementType>
class LinkedStack : public Stack<ElementType>
{
private:
	struct StackNode
	{
		ElementType data;
		StackNode* next;
		StackNode() : next(nullptr) {}
		StackNode(const ElementType& _data, StackNode* _next = nullptr) : data(_data), next(_next) {}
		~StackNode() {}
	};
	StackNode* head;

public:
	LinkedStack();
	~LinkedStack();
	bool empty() const;
	void push(const ElementType& element);
	ElementType pop();
	ElementType top() const;
	void clear();
};

template <typename ElementType>
LinkedStack<ElementType>::LinkedStack()
{
	head = nullptr;
}

template <typename ElementType>
LinkedStack<ElementType>::~LinkedStack()
{
	clear();
}

template <typename ElementType>
bool LinkedStack<ElementType>::empty() const
{
	return head == nullptr;
}

template <typename ElementType>
void LinkedStack<ElementType>::push(const ElementType& element)
{
	head = new StackNode(element, head);
}

template <typename ElementType>
ElementType LinkedStack<ElementType>::pop()
{
	if (head == nullptr)
		throw EmptyContainer("Error: Stack is already empty");
	StackNode* temp = head;
	ElementType value = temp->data;
	head = head->next;
	delete temp;
	return value;
}

template <typename ElementType>
ElementType LinkedStack<ElementType>::top() const
{
	if (head == nullptr)
		throw EmptyContainer("Error: Stack is already empty");
	return head->data;
}

template <typename ElementType>
void LinkedStack<ElementType>::clear()
{
	StackNode* temp;
	while (head != nullptr)
	{
		temp = head;
		head = head->next;
		delete temp;
	}
}