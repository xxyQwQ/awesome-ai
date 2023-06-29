#pragma once
#include <iostream>
#include "Exception.h"
#include "String.h"

class SequentialString : public String
{
	friend SequentialString operator+(const SequentialString& x, const SequentialString& y);
	friend bool operator==(const SequentialString& x, const SequentialString& y);
	friend bool operator!=(const SequentialString& x, const SequentialString& y);
	friend bool operator>(const SequentialString& x, const SequentialString& y);
	friend bool operator>=(const SequentialString& x, const SequentialString& y);
	friend bool operator<(const SequentialString& x, const SequentialString& y);
	friend bool operator<=(const SequentialString& x, const SequentialString& y);
	friend std::istream& operator>>(std::istream& s, SequentialString& x);
	friend std::ostream& operator<<(std::ostream& s, const SequentialString& x);

private:
	static int BUFFER;
	char* data;
	int size;

public:
	SequentialString(const char* s = "");
	SequentialString(const SequentialString& s);
	~SequentialString();
	SequentialString& operator=(const SequentialString& s);
	SequentialString& operator+=(const SequentialString& s);
	int length() const;
	SequentialString slice(int p, int n) const;
	void insert(int p, const SequentialString& s);
	void remove(int p, int n);
	void reset(int n); // Reset buffer size for input stream
	char& operator[](int p);
	char& operator[](int p) const;
	int find(const SequentialString& s) const; // Match substring by KMP algorithm
};

int SequentialString::BUFFER = 128;

SequentialString::SequentialString(const char* s)
{
	for (size = 0; s[size] != '\0'; size++)
		;
	data = new char[size + 1];
	for (int i = 0; i < size; i++)
		data[i] = s[i];
	data[size] = '\0';
}

SequentialString::SequentialString(const SequentialString& s)
{
	size = s.size;
	data = new char[size + 1];
	for (int i = 0; i <= size; i++)
		data[i] = s.data[i];
}

SequentialString::~SequentialString()
{
	delete[] data;
}

SequentialString& SequentialString::operator=(const SequentialString& s)
{
	if (this == &s)
		return *this;
	delete[] data;
	size = s.size;
	data = new char[size + 1];
	for (int i = 0; i <= size; i++)
		data[i] = s.data[i];
	return *this;
}

SequentialString& SequentialString::operator+=(const SequentialString& s)
{
	return (*this) = (*this) + s;
}

int SequentialString::length() const
{
	return size;
}

SequentialString SequentialString::slice(int p, int n) const
{
	if (p < 0 || p >= size || n <= 0)
		return SequentialString("");
	SequentialString temp;
	if (p + n > size)
		temp.size = size - p;
	else
		temp.size = n;
	delete[] temp.data;
	temp.data = new char[temp.size + 1];
	for (int i = 0; i < temp.size; i++)
		temp.data[i] = data[p + i];
	temp.data[temp.size] = '\0';
	return temp;
}

void SequentialString::insert(int p, const SequentialString& s)
{
	if (p < 0 || p > size)
		return;
	char* temp = data;
	size += s.size;
	data = new char[size + 1];
	for (int i = 0; i < p; i++)
		data[i] = temp[i];
	for (int i = 0; i < s.size; i++)
		data[p + i] = s.data[i];
	for (int i = p; temp[i] != '\0'; i++)
		data[s.size + i] = temp[i];
	data[size] = '\0';
	delete[] temp;
}

void SequentialString::remove(int p, int n)
{
	if (p < 0 || p >= size || n <= 0)
		return;
	if (p + n >= size)
	{
		data[p] = '\0';
		size = p;
	}
	else
	{
		for (size = p; data[size + n] != '\0'; size++)
			data[size] = data[size + n];
		data[size] = '\0';
	}
}

void SequentialString::reset(int n)
{
	if (n <= 0)
		return;
	BUFFER = n;
}

char& SequentialString::operator[](int p)
{
	if (p < 0 || p >= size)
		throw IndexExceed();
	return data[p];
}

char& SequentialString::operator[](int p) const
{
	if (p < 0 || p >= size)
		throw IndexExceed();
	return data[p];
}

SequentialString operator+(const SequentialString& x, const SequentialString& y)
{
	SequentialString temp(x);
	temp.insert(temp.size, y);
	return temp;
}

bool operator==(const SequentialString& x, const SequentialString& y)
{
	if (x.size != y.size)
		return false;
	for (int i = 0; i < x.size; i++)
		if (x.data[i] != y.data[i])
			return false;
	return true;
}

bool operator!=(const SequentialString& x, const SequentialString& y)
{
	return !(x == y);
}

bool operator>(const SequentialString& x, const SequentialString& y)
{
	for (int i = 0; i < x.size; i++)
		if (x.data[i] > y.data[i])
			return true;
		else if (x.data[i] < y.data[i])
			return false;
	return false;
}

bool operator>=(const SequentialString& x, const SequentialString& y)
{
	return (x == y || x > y);
}

bool operator<(const SequentialString& x, const SequentialString& y)
{
	return !(x >= y);
}

bool operator<=(const SequentialString& x, const SequentialString& y)
{
	return !(x > y);
}

std::istream& operator>>(std::istream& s, SequentialString& x)
{
	char* v = new char[SequentialString::BUFFER];
	s >> v;
	x = SequentialString(v);
	delete[] v;
	return s;
}

std::ostream& operator<<(std::ostream& s, const SequentialString& x)
{
	s << x.data;
	return s;
}

int SequentialString::find(const SequentialString& s) const
{
	int* p = new int[s.size];
	p[0] = -1;
	for (int i = 1, j; i < s.size; i++)
	{
		j = i - 1;
		while (j >= 0 && s.data[p[j] + 1] != s.data[i])
			j = p[j];
		if (j < 0)
			p[i] = -1;
		else
			p[i] = p[j] + 1;
	}
	int i = 0, j = 0;
	while (i < size && j < s.size)
	{
		if (data[i] == s.data[j])
		{
			i++;
			j++;
		}
		else if (j == 0)
			i++;
		else
			j = p[j - 1] + 1;
	}
	delete[] p;
	if (j == s.size)
		return i - j;
	else
		return -1;
}