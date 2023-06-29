#pragma once
#include "Exception.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class CloseHashTable : public DynamicSet<KeyType, DataType>
{
private:
	struct HashNode
	{
		SetElement<KeyType, DataType> data;
		int state; // 0 for empty, 1 for active, 2 for deleted
		HashNode() : data(), state(0) {}
		HashNode(const SetElement<KeyType, DataType>& _data, int _state = 0) : data(_data), state(_state) {}
	};
	HashNode* table;
	int size;
	int (*parse)(const KeyType& key); // function pointer: parse key to integer
	static int _parse(const int& key); // default parse function
public:
	CloseHashTable(int quantity = 101, int (*function)(const KeyType& key) = _parse);
	~CloseHashTable();
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
};

template<typename KeyType, typename DataType>
inline int CloseHashTable<KeyType, DataType>::_parse(const int& key)
{
	return key;
}

template<typename KeyType, typename DataType>
inline CloseHashTable<KeyType, DataType>::CloseHashTable(int quantity, int(*function)(const KeyType& key))
{
	size = quantity;
	table = new HashNode[size];
	parse = function;
}

template<typename KeyType, typename DataType>
inline CloseHashTable<KeyType, DataType>::~CloseHashTable()
{
	delete[] table;
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* CloseHashTable<KeyType, DataType>::find(const KeyType& key) const
{
	int origin, current;
	origin = current = parse(key) % size;
	do
	{
		if (table[current].state == 0)
			return nullptr;
		if (table[current].state == 1 && table[current].data.key == key)
			return &(table[current].data);
		current = (current + 1) % size;
	} while (current != origin);
	return nullptr;
}

template<typename KeyType, typename DataType>
inline void CloseHashTable<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	int origin, current;
	origin = current = parse(element.key) % size;
	do
	{
		if (table[current].state != 1)
		{
			table[current].data = element;
			table[current].state = 1;
			return;
		}
		current = (current + 1) % size;
	} while (current != origin);
}

template<typename KeyType, typename DataType>
inline void CloseHashTable<KeyType, DataType>::remove(const KeyType& key)
{
	int origin, current;
	origin = current = parse(key) % size;
	do
	{
		if (table[current].state == 0)
			return;
		if (table[current].state == 1 && table[current].data.key == key)
		{
			table[current].state = 2;
			return;
		}
		current = (current + 1) % size;
	} while (current != origin);
}