#pragma once
#include "Exception.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class OpenHashTable : public DynamicSet<KeyType, DataType>
{
private:
	struct HashNode
	{
		SetElement<KeyType, DataType> data;
		HashNode* next;
		HashNode() : data(), next(nullptr) {}
		HashNode(const SetElement<KeyType, DataType>& _data, HashNode* _next = nullptr) : data(_data), next(_next) {}
	};
	HashNode** table;
	int size;
	int (*parse)(const KeyType& key); // function pointer: parse key to integer
	static int _parse(const int& key); // default parse function
public:
	OpenHashTable(int quantity = 101, int (*function)(const KeyType& key) = _parse);
	~OpenHashTable();
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
};

template<typename KeyType, typename DataType>
inline int OpenHashTable<KeyType, DataType>::_parse(const int& key)
{
	return key;
}

template<typename KeyType, typename DataType>
inline OpenHashTable<KeyType, DataType>::OpenHashTable(int quantity, int(*function)(const KeyType& key))
{
	size = quantity;
	table = new HashNode * [size];
	parse = function;
	for (int i = 0; i < size; i++)
		table[i] = nullptr;
}

template<typename KeyType, typename DataType>
inline OpenHashTable<KeyType, DataType>::~OpenHashTable()
{
	HashNode* current = nullptr, * temp = nullptr;
	for (int i = 0; i < size; i++)
	{
		current = table[i];
		while (current != nullptr)
		{
			temp = current->next;
			delete current;
			current = temp;
		}
	}
	delete[] table;
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* OpenHashTable<KeyType, DataType>::find(const KeyType& key) const
{
	int position = parse(key) % size;
	HashNode* current = table[position];
	while (current != nullptr && (current->data).key != key)
		current = current->next;
	if (current == nullptr)
		return nullptr;
	else
		return &(current->data);
}

template<typename KeyType, typename DataType>
inline void OpenHashTable<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	int position = parse(element.key) % size;
	table[position] = new HashNode(element, table[position]);
}

template<typename KeyType, typename DataType>
inline void OpenHashTable<KeyType, DataType>::remove(const KeyType& key)
{
	int position = parse(key) % size;
	HashNode* current = table[position], * temp = nullptr;
	if (current == nullptr)
		return;
	if ((current->data).key == key)
	{
		table[position] = current->next;
		delete current;
		return;
	}
	while (current->next != nullptr && (current->next->data).key != key)
		current = current->next;
	if (current->next != nullptr)
	{
		temp = current->next;
		current->next = temp->next;
		delete temp;
	}
}