#pragma once

template <typename ElementType>
class Stack
{
public:
	virtual ~Stack() {}
	virtual bool empty() const = 0;
	virtual void push(const ElementType& element) = 0;
	virtual ElementType pop() = 0;
	virtual ElementType top() const = 0;
	virtual void clear() = 0;
};