#pragma once
#include <iostream>
#include "Exception.h"
#include "SequentialString.h"
#include "PriorityQueue.h"

template <class ElementType>
class HuffmanTree
{
private:
	struct HuffmanNode
	{
		ElementType data;
		int weight, index, parent, left, right;
		HuffmanNode() : data(), weight(0), index(0), parent(0), left(0), right(0) {}
		HuffmanNode(const ElementType& _data, int _weight, int _index = 0, int _parent = 0, int _left = 0, int _right = 0) : data(_data), weight(_weight), index(_index), parent(_parent), left(_left), right(_right) {}
		bool operator<(const HuffmanNode& another) const
		{
			return weight < another.weight;
		}
	};
	HuffmanNode* elementData;
	int totalLength, totalWeight;

public:
	struct HuffmanCode
	{
		ElementType data;
		SequentialString code;
		HuffmanCode() : data(), code() {}
		HuffmanCode(ElementType _data, SequentialString _code = "") : data(_data), code(_code) {}
	};
	HuffmanTree(const ElementType* content, const int* frequency, int size);
	~HuffmanTree();
	int getWeight() const;
	void getCode(HuffmanCode result[]) const;
};

template <typename ElementType>
HuffmanTree<ElementType>::HuffmanTree(const ElementType* content, const int* frequency, int size)
{
	totalLength = 2 * size, totalWeight = 0;
	elementData = new HuffmanNode[totalLength];
	PriorityQueue<HuffmanNode> heap;
	for (int i = size; i < totalLength; i++)
	{
		elementData[i] = HuffmanNode(content[i - size], frequency[i - size], i);
		heap.push(elementData[i]);
	}
	PriorityQueue<int> Heap;
	for (int i = size - 1; i; i--)
	{
		HuffmanNode first = heap.pop(), second = heap.pop();
		elementData[i] = HuffmanNode(first.data, first.weight + second.weight, i, 0, first.index, second.index);
		heap.push(elementData[i]);
		totalWeight += elementData[i].weight;
		elementData[first.index].parent = elementData[second.index].parent = i;
	}
}

template <typename ElementType>
HuffmanTree<ElementType>::~HuffmanTree()
{
	delete[] elementData;
}

template <typename ElementType>
int HuffmanTree<ElementType>::getWeight() const
{
	return totalWeight;
}

template <typename ElementType>
void HuffmanTree<ElementType>::getCode(HuffmanCode result[]) const
{
	int size = totalLength / 2;
	for (int i = size, current, father; i < totalLength; i++)
	{
		result[i - size] = HuffmanCode(elementData[i].data);
		current = i, father = elementData[current].parent;
		while (father)
		{
			if (elementData[father].left == current)
				result[i - size].code = "0" + result[i - size].code;
			else
				result[i - size].code = "1" + result[i - size].code;
			current = father, father = elementData[current].parent;
		}
	}
}