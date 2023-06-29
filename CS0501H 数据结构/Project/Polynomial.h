#pragma once
#include <iostream>

class Polynomial
{
	friend Polynomial addition(const Polynomial& x, const Polynomial& y);
	friend Polynomial operator+(const Polynomial& x, const Polynomial& y);

private:
	struct Item
	{
		int coef, expo;
		Item* next;
		Item() : coef(0), expo(0), next(nullptr) {}
		Item(int _coef, int _expo, Item* _next = nullptr) : coef(_coef), expo(_expo), next(_next) {}
	};
	Item* head;

public:
	Polynomial();
	Polynomial(const Polynomial& another);
	~Polynomial();
	void clear();
	void input();
	void output() const;
};

Polynomial::Polynomial()
{
	head = new Item;
}

Polynomial::Polynomial(const Polynomial& another)
{
	head = new Item;
	Item* p = another.head->next, * q = head;
	while (p != nullptr)
	{
		q = q->next = new Item(p->coef, p->expo);
		p = p->next;
	}
}

Polynomial::~Polynomial()
{
	clear();
	delete head;
}

void Polynomial::clear()
{
	Item* p = head->next, * q;
	head->next = nullptr;
	while (p != nullptr)
	{
		q = p->next;
		delete p;
		p = q;
	}
}

void Polynomial::input()
{
	clear();
	Item* p = head;
	int x, y;
	std::cout << "Enter \"-1 -1\" to finish input.\n";
	std::cout << "Input coeffience and exponent: ";
	std::cin >> x >> y;
	while (x != -1 || y != -1)
	{
		if (x != 0)
			p = p->next = new Item(x, y);
		std::cout << "Input coeffience and exponent: ";
		std::cin >> x >> y;
	}
	std::cout << "Polynomial input finished!\n";
}

void Polynomial::output() const
{
	Item* p = head->next;
	if (p == nullptr)
	{
		std::cout << "0\n";
		return;
	}
	std::cout << p->coef << "x^" << p->expo;
	p = p->next;
	while (p != nullptr)
	{
		if (p->coef > 0)
			std::cout << "+";
		std::cout << p->coef << "x^" << p->expo;
		p = p->next;
	}
	std::cout << "\n";
}

Polynomial addition(const Polynomial& x, const Polynomial& y)
{
	Polynomial r;
	Polynomial::Item* u = x.head->next, * v = y.head->next, * p = r.head;
	while (u != nullptr && v != nullptr)
	{
		if (u->expo < v->expo)
		{
			p = p->next = new Polynomial::Item(u->coef, u->expo);
			u = u->next;
		}
		else if (u->expo > v->expo)
		{
			p = p->next = new Polynomial::Item(v->coef, v->expo);
			v = v->next;
		}
		else if (u->coef + v->coef != 0)
		{
			p = p->next = new Polynomial::Item(u->coef + v->coef, u->expo);
			u = u->next, v = v->next;
		}
		else
			u = u->next, v = v->next;
	}
	Polynomial::Item* q = u == nullptr ? v : u;
	while (q != nullptr)
	{
		p = p->next = new Polynomial::Item(q->coef, q->expo);
		q = q->next;
	}
	return r;
}

Polynomial operator+(const Polynomial& x, const Polynomial& y)
{
	return addition(x, y);
}