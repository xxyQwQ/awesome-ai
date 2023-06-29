#pragma once
#include <iostream>
#include <string>
#include <cstdlib>
#include "Exception.h"
#include "SequentialList.h"

class BigInteger : public SequentialList<int>
{
	friend std::istream& operator>>(std::istream& is, BigInteger& x);
	friend std::ostream& operator<<(std::ostream& os, const BigInteger& x);
	friend bool operator==(const BigInteger& x, const BigInteger& y);
	friend bool operator!=(const BigInteger& x, const BigInteger& y);
	friend bool operator<(const BigInteger& x, const BigInteger& y);
	friend bool operator>(const BigInteger& x, const BigInteger& y);
	friend bool operator<=(const BigInteger& x, const BigInteger& y);
	friend bool operator>=(const BigInteger& x, const BigInteger& y);
	friend BigInteger absolute(const BigInteger& x);
	friend BigInteger addition(const BigInteger& x, const BigInteger& y);       // addition of absolute value
	friend BigInteger subtraction(const BigInteger& x, const BigInteger& y);    // subtraction of absolute value
	friend BigInteger multiplication(const BigInteger& x, const BigInteger& y); // multiplication of absolute value
	friend BigInteger division(const BigInteger& x, const BigInteger& y);       // division of absolute value
	friend BigInteger power(const BigInteger& x, const BigInteger& y);          // y power of x
	friend BigInteger operator-(const BigInteger& x);
	friend BigInteger operator+(const BigInteger& x, const BigInteger& y);
	friend BigInteger operator-(const BigInteger& x, const BigInteger& y);
	friend BigInteger operator*(const BigInteger& x, const BigInteger& y);
	friend BigInteger operator/(const BigInteger& x, const BigInteger& y);
	friend BigInteger operator%(const BigInteger& x, const BigInteger& y);

private:
	bool sign; // 0 for positive, 1 for negative
	BigInteger& update();
	void construct(const std::string& s);

public:
	BigInteger(int n = 0);
	BigInteger(const std::string& s);
	BigInteger(const BigInteger& x);
	~BigInteger();
	BigInteger& operator=(const BigInteger& x);
	BigInteger& operator+=(const BigInteger& x);
	BigInteger& operator-=(const BigInteger& x);
	BigInteger& operator*=(const BigInteger& x);
	BigInteger& operator/=(const BigInteger& x);
	BigInteger& operator%=(const BigInteger& x);
};

BigInteger::BigInteger(int n)
{
	if (n >= 0)
	{
		sign = 0;
		append(n);
	}
	else
	{
		sign = 1;
		append(-n);
	}
	update();
}

BigInteger::BigInteger(const std::string& s)
{
	if (s == "")
		construct("0");
	else
		construct(s);
}

BigInteger::BigInteger(const BigInteger& x)
{
	sign = x.sign;
	for (int i = 0; i < x.length(); i++)
		append(x[i]);
}

BigInteger::~BigInteger()
{
	clear();
}

BigInteger& BigInteger::operator=(const BigInteger& x)
{
	if (this == &x)
		return *this;
	clear();
	sign = x.sign;
	for (int i = 0; i < x.length(); i++)
		append(x[i]);
	return *this;
}

BigInteger& BigInteger::update()
{
	while (length() > 0 && back() == 0)
		cancel();
	if (length() == 0)
	{
		sign = 0;
		return *this;
	}
	for (int i = 1; i < length(); i++)
	{
		(*this)[i] += (*this)[i - 1] / 10;
		(*this)[i - 1] %= 10;
	}
	while (back() >= 10)
	{
		append(back() / 10);
		(*this)[length() - 2] %= 10;
	}
	return *this;
}

void BigInteger::construct(const std::string& s)
{
	int p = s.length(), q;
	if (isdigit(s[0]))
	{
		q = 0;
		sign = 0;
	}
	else
	{
		q = 1;
		sign = (s[0] == '-');
	}
	for (int i = p - 1; i >= q; i--)
		append(s[i] - '0');
	update();
}

std::istream& operator>>(std::istream& is, BigInteger& x)
{
	std::string s;
	is >> s;
	x.clear();
	x.construct(s);
	return is;
}

std::ostream& operator<<(std::ostream& os, const BigInteger& x)
{
	if (x.length() == 0 || x == 0)
		os << 0;
	if (x.sign == 1)
		os << "-";
	for (int i = x.length() - 1; i >= 0; i--)
		os << x[i];
	return os;
}

bool operator!=(const BigInteger& x, const BigInteger& y)
{
	if (x.sign != y.sign)
		return true;
	if (x.length() != y.length())
		return true;
	for (int i = 0; i < x.length(); i++)
		if (x[i] != y[i])
			return true;
	return false;
}

bool operator==(const BigInteger& x, const BigInteger& y)
{
	return !(x != y);
}

bool operator<(const BigInteger& x, const BigInteger& y)
{
	if (x.sign == 0 && y.sign == 1)
		return false;
	if (x.sign == 1 && y.sign == 0)
		return true;
	if (x.length() != y.length())
		return x.length() < y.length();
	for (int i = x.length() - 1; i >= 0; i--)
		if (x[i] != y[i])
			return x[i] < y[i];
	return false;
}

bool operator>(const BigInteger& x, const BigInteger& y)
{
	return y < x;
}

bool operator<=(const BigInteger& x, const BigInteger& y)
{
	return !(x > y);
}

bool operator>=(const BigInteger& x, const BigInteger& y)
{
	return !(x < y);
}

BigInteger absolute(const BigInteger& x)
{
	if (x.sign == 0)
		return x;
	else
		return -x;
}

BigInteger addition(const BigInteger& x, const BigInteger& y)
{
	BigInteger r = x; // Suppose x, y are both positive
	while (r.length() < y.length())
		r.append(0);
	for (int i = 0; i < y.length(); i++)
		r[i] += y[i];
	r.sign = 0;
	return r.update();
}

BigInteger subtraction(const BigInteger& x, const BigInteger& y)
{
	BigInteger r = x; // It is promised that x > y
	for (int i = 0; i < y.length(); i++)
	{
		if (r[i] < y[i])
		{
			int j = i + 1;
			while (r[j] == 0)
				j++;
			while (j > i)
			{
				r[j--]--;
				r[j] += 10;
			}
		}
		r[i] -= y[i];
	}
	r.sign = 0;
	return r.update();
}

BigInteger multiplication(const BigInteger& x, const BigInteger& y)
{
	BigInteger r; // Suppose x, y are both positive
	for (int i = 0; i < x.length() + y.length(); i++)
		r.append(0);
	for (int i = 0; i < x.length(); i++)
		for (int j = 0; j < y.length(); j++)
			r[i + j] += x[i] * y[j];
	r.sign = 0;
	return r.update();
}

BigInteger division(const BigInteger& x, const BigInteger& y)
{
	BigInteger z(x), r(0), u, v; // Suppose x, y are both positive and y is not 0
	for (int i = x.length() - y.length(); z >= y; i--)
	{
		u.clear();
		for (int j = 0; j < i; j++)
			u.append(0);
		u.append(1);
		v = y * u;
		while (z >= v)
		{
			z -= v;
			r += u;
		}
	}
	return r.update();
}

BigInteger power(const BigInteger& x, const BigInteger& y)
{
	if (y == 0)
		return BigInteger(1);
	BigInteger r(1), a(x), k(y);
	while (k != 0)
	{
		if (k % 2 == 1)
			r *= a;
		a *= a;
		k /= 2;
	}
	return r;
}

BigInteger operator-(const BigInteger& x)
{
	BigInteger r = x;
	r.sign = !r.sign;
	return r;
}

BigInteger& BigInteger::operator+=(const BigInteger& x)
{
	return (*this) = (*this) + x;
}

BigInteger& BigInteger::operator-=(const BigInteger& x)
{
	return (*this) = (*this) - x;
}

BigInteger& BigInteger::operator*=(const BigInteger& x)
{
	return (*this) = (*this) * x;
}

BigInteger& BigInteger::operator/=(const BigInteger& x)
{
	return (*this) = (*this) / x;
}

BigInteger& BigInteger::operator%=(const BigInteger& x)
{
	return (*this) = (*this) % x;
}

BigInteger operator+(const BigInteger& x, const BigInteger& y)
{
	BigInteger r;
	if (x.sign == 0 && y.sign == 0)
	{
		r = addition(x, y);
		r.sign = 0;
	}
	else if (x.sign == 1 && y.sign == 1)
	{
		r = addition(x, y);
		r.sign = 1;
	}
	else if (x.sign == 0 && y.sign == 1)
	{
		if (x >= absolute(y))
		{
			r = subtraction(x, y);
			r.sign = 0;
		}
		else
		{
			r = subtraction(y, x);
			r.sign = 1;
		}
	}
	else if (x.sign == 1 && y.sign == 0)
	{
		if (absolute(x) <= y)
		{
			r = subtraction(y, x);
			r.sign = 0;
		}
		else
		{
			r = subtraction(x, y);
			r.sign = 1;
		}
	}
	return r.update();
}

BigInteger operator-(const BigInteger& x, const BigInteger& y)
{
	return x + (-y);
}

BigInteger operator*(const BigInteger& x, const BigInteger& y)
{
	BigInteger r = multiplication(x, y);
	if (x.sign == y.sign)
		r.sign = 0;
	else
		r.sign = 1;
	return r.update();
}

BigInteger operator/(const BigInteger& x, const BigInteger& y)
{
	if (y == 0)
		throw DivideByZero();
	BigInteger r = division(absolute(x), absolute(y));
	if (x.sign == y.sign)
		r.sign = 0;
	else
		r.sign = 1;
	return r.update();
}

BigInteger operator%(const BigInteger& x, const BigInteger& y)
{
	return x - (y * (x / y));
}