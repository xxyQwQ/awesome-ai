#pragma once
#include "Exception.h"

class FindUnionSet
{
private:
	int* parent;
	int size;
public:
	FindUnionSet(int n);
	~FindUnionSet();
	int find(int x);
	void merge(int x, int y);
};

FindUnionSet::FindUnionSet(int n)
{
	size = n;
	parent = new int[size];
	for (int i = 0; i < size; i++)
		parent[i] = i;
}

FindUnionSet::~FindUnionSet()
{
	delete[] parent;
}

int FindUnionSet::find(int x)
{
	if (parent[x] == x)
		return x;
	else
		return parent[x] = find(parent[x]);
}

void FindUnionSet::merge(int x, int y)
{
	int p = find(x), q = find(y);
	if (p != q)
		parent[p] = q;
}