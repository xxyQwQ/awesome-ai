#pragma once
#include "Exception.h"
#include "Queue.h"

template <typename ElementType>
class LinkedQueue : public Queue<ElementType>
{
private:
	struct QueueNode
	{
		ElementType data;
		QueueNode* next;
		QueueNode() : next(nullptr) {}
		QueueNode(const ElementType& _data, QueueNode* _next = nullptr) : data(_data), next(_next) {}
		~QueueNode() {}
	};
	QueueNode* head, * tail;

public:
	LinkedQueue();
	~LinkedQueue();
	bool empty() const;
	void push(const ElementType& element);
	ElementType pop();
	ElementType front() const;
	void clear();
};

template <typename ElementType>
LinkedQueue<ElementType>::LinkedQueue()
{
	head = tail = nullptr;
}

template <typename ElementType>
LinkedQueue<ElementType>::~LinkedQueue()
{
	clear();
}

template <typename ElementType>
bool LinkedQueue<ElementType>::empty() const
{
	return head == nullptr;
}

template <typename ElementType>
void LinkedQueue<ElementType>::push(const ElementType& element)
{
	if (tail == nullptr)
		head = tail = new QueueNode(element);
	else
		tail = tail->next = new QueueNode(element);
}

template <typename ElementType>
ElementType LinkedQueue<ElementType>::pop()
{
	if (head == nullptr)
		throw EmptyContainer("Error: Queue is already empty");
	QueueNode* temp = head;
	ElementType value = temp->data;
	head = head->next;
	if (head == nullptr)
		tail = nullptr;
	delete temp;
	return value;
}

template <typename ElementType>
ElementType LinkedQueue<ElementType>::front() const
{
	if (head == nullptr)
		throw EmptyContainer("Error: Queue is already empty");
	return head->data;
}

template <typename ElementType>
void LinkedQueue<ElementType>::clear()
{
	QueueNode* temp;
	while (head != nullptr)
	{
		temp = head;
		head = head->next;
		delete temp;
	}
}