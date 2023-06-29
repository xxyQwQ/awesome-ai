#pragma once
#include "Set.h"

template <typename KeyType, typename DataType>
class DynamicSet
{
public:
	virtual const SetElement<KeyType, DataType>* find(const KeyType& key) const = 0;
	virtual void insert(const SetElement<KeyType, DataType>& element) = 0;
	virtual void remove(const KeyType& key) = 0;
	virtual ~DynamicSet() {}
};