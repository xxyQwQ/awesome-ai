#pragma once

template <typename ElementType>
class List
{
public:
	virtual ~List() {}
	virtual void clear() = 0;
	virtual int length() const = 0;
	virtual void insert(int index, const ElementType& element) = 0;
	virtual void remove(int index) = 0;
	virtual int find(const ElementType& element) const = 0;
	virtual ElementType fetch(int index) const = 0;
	virtual void traverse() const = 0;
};