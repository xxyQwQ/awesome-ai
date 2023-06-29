#pragma once
#include "Exception.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class BinarySearchTree : public DynamicSet<KeyType, DataType>
{
private:
	struct TreeNode
	{
		SetElement<KeyType, DataType> data;
		TreeNode* left, * right;
		TreeNode() :data(), left(nullptr), right(nullptr) {}
		TreeNode(const SetElement<KeyType, DataType>& _data, TreeNode* _left = nullptr, TreeNode* _right = nullptr) :data(_data), left(_left), right(_right) {}
	};
	TreeNode* root;
	const SetElement<KeyType, DataType>* find(TreeNode* current, const KeyType& key) const;
	void insert(TreeNode*& current, const SetElement<KeyType, DataType>& element);
	void remove(TreeNode*& current, const KeyType& key);
	void clear(TreeNode* current);
public:
	BinarySearchTree();
	~BinarySearchTree();
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
	void clear();
};

template <typename KeyType, typename DataType>
const SetElement<KeyType, DataType>* BinarySearchTree<KeyType, DataType>::find(TreeNode* current, const KeyType& key) const
{
	if (current == nullptr)
		return nullptr;
	if (key == (current->data).key)
		return &(current->data);
	if (key < (current->data).key)
		return find(current->left, key);
	else
		return find(current->right, key);
}

template <typename KeyType, typename DataType>
void BinarySearchTree<KeyType, DataType>::insert(TreeNode*& current, const SetElement<KeyType, DataType>& element)
{
	if (current == nullptr)
		current = new TreeNode(element);
	else if (element.key < (current->data).key)
		insert(current->left, element);
	else if (element.key > (current->data).key)
		insert(current->right, element);
}

template <typename KeyType, typename DataType>
void BinarySearchTree<KeyType, DataType>::remove(TreeNode*& current, const KeyType& key)
{
	if (current == nullptr)
		return;
	if (key < (current->data).key)
		remove(current->left, key);
	else if (key > (current->data).key)
		remove(current->right, key);
	else if (current->left != nullptr && current->right != nullptr)
	{
		TreeNode* temp = current->right;
		while (temp->left != nullptr)
			temp = temp->left;
		current->data = temp->data;
		remove(current->right, (current->data).key);
	}
	else
	{
		TreeNode* temp = current;
		if (current->left != nullptr)
			current = current->left;
		else
			current = current->right;
		delete temp;
	}
}

template <typename KeyType, typename DataType>
void BinarySearchTree<KeyType, DataType>::clear(TreeNode* current)
{
	if (current == nullptr)
		return;
	clear(current->left);
	clear(current->right);
	delete current;
}

template <typename KeyType, typename DataType>
BinarySearchTree<KeyType, DataType>::BinarySearchTree()
{
	root = nullptr;
}

template <typename KeyType, typename DataType>
BinarySearchTree<KeyType, DataType>::~BinarySearchTree()
{
	clear(root);
	root = nullptr;
}

template <typename KeyType, typename DataType>
const SetElement<KeyType, DataType>* BinarySearchTree<KeyType, DataType>::find(const KeyType& key) const
{
	return find(root, key);
}

template <typename KeyType, typename DataType>
void BinarySearchTree<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	insert(root, element);
}

template <typename KeyType, typename DataType>
void BinarySearchTree<KeyType, DataType>::remove(const KeyType& key)
{
	remove(root, key);
}

template <typename KeyType, typename DataType>
void BinarySearchTree<KeyType, DataType>::clear()
{
	clear(root);
	root = nullptr;
}