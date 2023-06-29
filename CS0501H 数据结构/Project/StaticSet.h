#pragma once
#include "Exception.h"
#include "Set.h"

template <typename KeyType, typename DataType>
int SequentialSearch(SetElement<KeyType, DataType> table[], int size, const KeyType& key) // return index of given key in unordered table by sequential search
{
	int i;
	table[0].key = key;
	for (i = size; table[i].key != key; i--);
	return i; // return 0 if key is not found
}

template <typename KeyType, typename DataType>
int BinarySearch(SetElement<KeyType, DataType> table[], int size, const KeyType& key) // return index of given key in ordered table by binary search
{
	int l = 1, r = size, h;
	while (l <= r)
	{
		h = (l + r) >> 1;
		if (table[h].key == key)
			return h;
		if (table[h].key > key)
			r = h - 1;
		else
			l = h + 1;
	}
	return 0; // return 0 if key is not found
}