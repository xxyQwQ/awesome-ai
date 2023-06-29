#pragma once

class String
{
public:
	virtual ~String() {}
	virtual int length() const = 0;
	virtual char& operator[](int i) = 0;
	virtual char& operator[](int i) const = 0;
};