#pragma once
#include <iostream>
#include "Exception.h"
#include "Function.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class AATree : public DynamicSet<KeyType, DataType>
{
private:
	struct TreeNode
	{
		SetElement<KeyType, DataType> data;
		TreeNode* left, * right;
		int level;
		TreeNode() :data(), left(nullptr), right(nullptr), level(1) {}
		TreeNode(const SetElement<KeyType, DataType>& _data, TreeNode* _left = nullptr, TreeNode* _right = nullptr, int _level = 1) :data(_data), left(_left), right(_right), level(_level) {}
	};
	TreeNode* root;
	void right_rotate(TreeNode*& current); // LL
	void left_rotate(TreeNode*& current); // RR
	void tree_traverse(TreeNode* current) const; // preorder
	void insert_adjust(TreeNode*& current, const SetElement<KeyType, DataType>& element);
	void remove_adjust(TreeNode*& current, const KeyType& key);
	void make_empty(TreeNode*& current);
public:
	AATree();
	~AATree();
	void traverse() const;
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
	void clear();
};

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::right_rotate(TreeNode*& current)
{
	if (current->left == nullptr || current->left->level != current->level)
		return;
	TreeNode* temp = current->left;
	current->left = temp->right;
	temp->right = current;
	current = temp;
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::left_rotate(TreeNode*& current)
{
	if (current->right == nullptr || current->right->right == nullptr || current->right->right->level != current->level)
		return;
	TreeNode* temp = current->right;
	current->right = temp->left;
	temp->left = current;
	current = temp;
	(current->level)++;
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::tree_traverse(TreeNode* current) const
{
	if (current == nullptr)
		return;
	std::cout << (current->data).key << "\t" << (current->data).data << "\t" << current->level << "\t";
	if (current->left == nullptr)
		std::cout << "L: *\t";
	else
		std::cout << "L: " << (current->left->data).key << "\t";
	if (current->right == nullptr)
		std::cout << "R: *\t";
	else
		std::cout << "R: " << (current->right->data).key << "\t";
	std::cout << "\n";
	tree_traverse(current->left);
	tree_traverse(current->right);
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::insert_adjust(TreeNode*& current, const SetElement<KeyType, DataType>& element)
{
	if (current == nullptr)
		current = new TreeNode(element);
	else if (element.key < (current->data).key)
		insert_adjust(current->left, element);
	else if (element.key > (current->data).key)
		insert_adjust(current->right, element);
	else
		return;
	right_rotate(current);
	left_rotate(current);
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::remove_adjust(TreeNode*& current, const KeyType& key)
{
	if (current == nullptr) return;
	if (key < (current->data).key)
		remove_adjust(current->left, key);
	else if (key > (current->data.key))
		remove_adjust(current->right, key);
	else if (current->left != nullptr && current->right != nullptr)
	{
		TreeNode* temp = current->right;
		while (temp->left != nullptr)
			temp = temp->left;
		current->data = temp->data;
		remove_adjust(current->right, (current->data).key);
	}
	else
	{
		TreeNode* temp = current;
		if (current->left != nullptr)
			current = current->left;
		else
			current = current->right;
		delete temp;
		return;
	}
	if (current->left == nullptr || current->right == nullptr)
		current->level = 1;
	else
		current->level = min(current->left->level, current->right->level) + 1;
	if (current->right != nullptr && current->right->level > current->level)
		current->right->level = current->level;
	right_rotate(current);
	if (current->right != nullptr)
	{
		right_rotate(current->right);
		if (current->right->right != nullptr)
			right_rotate(current->right->right);
	}
	left_rotate(current);
	if (current->right != nullptr)
		left_rotate(current->right);
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::make_empty(TreeNode*& current)
{
	if (current == nullptr)
		return;
	make_empty(current->left);
	make_empty(current->right);
	delete(current);
}

template<typename KeyType, typename DataType>
inline AATree<KeyType, DataType>::AATree()
{
	root = nullptr;
}

template<typename KeyType, typename DataType>
inline AATree<KeyType, DataType>::~AATree()
{
	make_empty(root);
	root = nullptr;
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::traverse() const
{
	std::cout << "Elements in set are as follow:\n";
	tree_traverse(root);
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* AATree<KeyType, DataType>::find(const KeyType& key) const
{
	TreeNode* current = root;
	while (current != nullptr && (current->data).key != key)
	{
		if ((current->data).key > key)
			current = current->left;
		else
			current = current->right;
	}
	if (current == nullptr)
		return nullptr;
	return &(current->data);
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	insert_adjust(root, element);
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::remove(const KeyType& key)
{
	remove_adjust(root, key);
}

template<typename KeyType, typename DataType>
inline void AATree<KeyType, DataType>::clear()
{
	make_empty(root);
	root = nullptr;
}