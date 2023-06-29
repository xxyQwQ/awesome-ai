#pragma once

template <typename ElementType>
class Queue
{
public:
	virtual ~Queue() {}
	virtual bool empty() const = 0;
	virtual void push(const ElementType& element) = 0;
	virtual ElementType pop() = 0;
	virtual ElementType front() const = 0;
	virtual void clear() = 0;
};