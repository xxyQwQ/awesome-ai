#pragma once
#include <iostream>
#include <cmath>
#include "Exception.h"
#include "String.h"

class LinkedString : public String
{
	friend LinkedString operator+(const LinkedString& x, const LinkedString& y);
	friend bool operator==(const LinkedString& x, const LinkedString& y);
	friend bool operator!=(const LinkedString& x, const LinkedString& y);
	friend bool operator>(const LinkedString& x, const LinkedString& y);
	friend bool operator>=(const LinkedString& x, const LinkedString& y);
	friend bool operator<(const LinkedString& x, const LinkedString& y);
	friend bool operator<=(const LinkedString& x, const LinkedString& y);
	friend std::istream& operator>>(std::istream& s, LinkedString& x);
	friend std::ostream& operator<<(std::ostream& s, const LinkedString& x);

private:
	static int BUFFER;
	struct StringNode
	{
		int size;
		char* data;
		StringNode* next;
		StringNode() : size(0), data(NULL), next(nullptr) {}
		StringNode(int _capacity = 1, StringNode* _next = nullptr) : size(0), next(_next)
		{
			data = new char[_capacity];
		}
		~StringNode()
		{
			delete data;
		}
	};
	StringNode* head;
	int size, capacity;
	void clear();
	void place(int target, int& index, StringNode*& block) const;
	void split(StringNode* block, int index);
	void merge(StringNode* block);

public:
	LinkedString(const char* origin = "");
	LinkedString(const LinkedString& another);
	LinkedString& operator=(const LinkedString& another);
	LinkedString& operator+=(const LinkedString& another);
	~LinkedString();
	int length() const;
	LinkedString slice(int start, int count);
	void insert(int start, const LinkedString& another);
	void remove(int start, int count);
	char& operator[](int i);
	char& operator[](int i) const;
};

int LinkedString::BUFFER = 128;

void LinkedString::clear()
{
	StringNode* current = head->next, * temp;
	while (current != nullptr)
	{
		temp = current->next;
		delete current;
		current = temp;
	}
	head->next = nullptr;
}

void LinkedString::place(int target, int& index, StringNode*& block) const
{
	int count = 0;
	index = 0;
	block = head->next;
	while (count < target)
	{
		if (count + block->size < target)
		{
			count += block->size;
			block = block->next;
		}
		else
		{
			index = target - count;
			return;
		}
	}
}

void LinkedString::split(StringNode* block, int index)
{
	block->next = new StringNode(capacity, block->next);
	for (int i = index; i < block->size; i++)
		block->next->data[i - index] = block->data[i];
	block->next->size = block->size - index;
	block->size = index;
}

void LinkedString::merge(StringNode* block)
{
	StringNode* correspond = block->next;
	if (correspond == nullptr || block->size + correspond->size > capacity)
		return;
	for (int i = 0; i < correspond->size; i++)
		block->data[block->size++] = correspond->data[i];
	block->next = correspond->next;
	delete correspond;
}

LinkedString::LinkedString(const char* origin)
{
	for (size = 0; origin[size] != '\0'; size++)
		;
	capacity = int(sqrt(size));
	StringNode* current;
	current = head = new StringNode(1);
	while (*origin)
	{
		current = current->next = new StringNode(capacity);
		while (*origin && current->size < capacity)
		{
			current->data[current->size] = *origin;
			current->size++;
			origin++;
		}
	}
}

LinkedString::LinkedString(const LinkedString& another)
{
	StringNode* current, * correspond = another.head->next;
	current = head = new StringNode(1);
	size = another.size;
	capacity = another.capacity;
	while (correspond != nullptr)
	{
		current = current->next = new StringNode(capacity);
		while (current->size < correspond->size)
		{
			current->data[current->size] = correspond->data[current->size];
			current->size++;
		}
		correspond = correspond->next;
	}
}

LinkedString& LinkedString::operator=(const LinkedString& another)
{
	if (this == &another)
		return *this;
	clear();
	StringNode* current = head, * correspond = another.head->next;
	size = another.size;
	capacity = another.capacity;
	while (correspond != nullptr)
	{
		current = current->next = new StringNode(capacity);
		while (current->size < correspond->size)
		{
			current->data[current->size] = correspond->data[current->size];
			current->size++;
		}
		correspond = correspond->next;
	}
	return *this;
}

LinkedString& LinkedString::operator+=(const LinkedString& another)
{
	return (*this) = (*this) + another;
}

LinkedString::~LinkedString()
{
	clear();
	delete head;
}

int LinkedString::length() const
{
	return size;
}

LinkedString LinkedString::slice(int start, int count)
{
	LinkedString temp;
	if (start < 0 || start >= size)
		return temp;
	if (start + count > size)
		count = size - start;
	temp.size = count;
	temp.capacity = int(sqrt(count));
	StringNode* current, * correspond = temp.head;
	int index;
	place(start, index, current);
	for (int i = 0; i < temp.size;)
	{
		correspond = correspond->next = new StringNode(temp.capacity);
		while (i < temp.size && correspond->size < temp.capacity)
		{
			if (index == current->size)
			{
				index = 0;
				current = current->next;
			}
			correspond->data[correspond->size] = current->data[index++];
			correspond->size++;
			i++;
		}
	}
	return temp;
}

void LinkedString::insert(int start, const LinkedString& another)
{
	if (start < 0 || start > size)
		return;
	StringNode* current, * correspond, * temp;
	int index;
	place(start, index, current);
	split(current, index);
	temp = current->next;
	correspond = another.head->next;
	while (correspond != nullptr)
	{
		for (index = 0; index < correspond->size; index++)
		{
			if (current->size == capacity)
				current = current->next = new StringNode(capacity);
			current->data[current->size++] = correspond->data[index];
		}
		correspond = correspond->next;
	}
	current->next = temp;
	size += another.size;
	merge(current);
}

void LinkedString::remove(int start, int count)
{
	if (start < 0 || start >= size)
		return;
	StringNode* current;
	int index;
	place(start, index, current);
	split(current, index);
	if (start + count >= size)
	{
		count = size - start;
		size = start;
	}
	else
		size -= count;
	while (true)
	{
		StringNode* temp = current->next;
		if (count > temp->size)
		{
			count -= temp->size;
			current->next = temp->next;
			delete temp;
		}
		else
		{
			split(temp, count);
			current->next = temp->next;
			delete temp;
			break;
		}
	}
	merge(current);
}

LinkedString operator+(const LinkedString& x, const LinkedString& y)
{
	LinkedString temp(x);
	temp.insert(temp.size, y);
	return temp;
}

bool operator==(const LinkedString& x, const LinkedString& y)
{
	if (x.size != y.size)
		return false;
	LinkedString::StringNode* px = x.head->next, * py = y.head->next;
	int nx = 0, ny = 0;
	while (px != nullptr && py != nullptr)
	{
		if (px->data[nx] != py->data[ny])
			return false;
		if (++nx == px->size)
		{
			px = px->next;
			nx = 0;
		}
		if (++ny == py->size)
		{
			py = py->next;
			ny = 0;
		}
	}
	return true;
}

bool operator!=(const LinkedString& x, const LinkedString& y)
{
	return !(x == y);
}

bool operator>(const LinkedString& x, const LinkedString& y)
{
	LinkedString::StringNode* px = x.head->next, * py = y.head->next;
	int nx = 0, ny = 0;
	while (px != nullptr)
	{
		if (py == nullptr)
			return true;
		if (px->data[nx] > py->data[ny])
			return true;
		if (px->data[nx] < py->data[ny])
			return false;
		if (++nx == px->size)
		{
			px = px->next;
			nx = 0;
		}
		if (++ny == py->size)
		{
			py = py->next;
			ny = 0;
		}
	}
	return false;
}

bool operator>=(const LinkedString& x, const LinkedString& y)
{
	return (x == y || x > y);
}

bool operator<(const LinkedString& x, const LinkedString& y)
{
	return !(x >= y);
}

bool operator<=(const LinkedString& x, const LinkedString& y)
{
	return !(x > y);
}

std::istream& operator>>(std::istream& s, LinkedString& x)
{
	char* v = new char[LinkedString::BUFFER];
	s >> v;
	x = LinkedString(v);
	delete[] v;
	return s;
}

std::ostream& operator<<(std::ostream& s, const LinkedString& x)
{
	LinkedString::StringNode* p = x.head->next;
	while (p != nullptr)
	{
		for (int i = 0; i < p->size; i++)
			s << p->data[i];
		p = p->next;
	}
	return s;
}

char& LinkedString::operator[](int i)
{
	if (i < 0 || i >= size)
		throw IndexExceed();
	LinkedString::StringNode* p;
	int n;
	place(i, n, p);
	if (n == p->size)
	{
		p = p->next;
		n = 0;
	}
	return p->data[n];
}

char& LinkedString::operator[](int i) const
{
	if (i < 0 || i >= size)
		throw IndexExceed();
	LinkedString::StringNode* p;
	int n;
	place(i, n, p);
	if (n == p->size)
	{
		p = p->next;
		n = 0;
	}
	return p->data[n];
}