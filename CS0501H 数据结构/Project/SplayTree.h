#pragma once
#include <iostream>
#include "Exception.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class SplayTree : public DynamicSet<KeyType, DataType>
{
private:
	struct TreeNode
	{
		SetElement<KeyType, DataType> data;
		int size;
		TreeNode* parent, * child[2]; // 0 for left child, 1 for right child
		TreeNode() : data(), size(0), parent(nullptr)
		{
			child[0] = child[1] = nullptr;
		}
		TreeNode(const SetElement<KeyType, DataType>& _data, int _size = 1, TreeNode* _parent = nullptr, TreeNode* _left = nullptr, TreeNode* _right = nullptr) : data(_data), size(_size), parent(_parent)
		{
			child[0] = _left, child[1] = _right;
		}
	} *root;
	void make_empty(TreeNode* current);
	bool which_child(TreeNode* parent, TreeNode* child); // 0 for left child, 1 for right child
	void connect_node(TreeNode* parent, TreeNode* child, bool which);
	void update_size(TreeNode* current);
	void rotate_node(TreeNode* current);
	void make_splay(TreeNode* current, TreeNode* target);
	void delete_node(int rank);
	TreeNode* rank_search(TreeNode* current, int rank);
	TreeNode* lower_bound(const KeyType& key);
	void tree_traverse(TreeNode* current) const; // preorder
public:
	SplayTree();
	~SplayTree();
	void clear();
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
	int rank(const KeyType& key);
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	const SetElement<KeyType, DataType>* fetch(int rank);
	const SetElement<KeyType, DataType>* precursor(const KeyType& key);
	const SetElement<KeyType, DataType>* successor(const KeyType& key);
};

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::make_empty(TreeNode* current)
{
	if (current == nullptr)
		return;
	make_empty(current->child[0]);
	make_empty(current->child[1]);
	delete current;
}

template<typename KeyType, typename DataType>
inline bool SplayTree<KeyType, DataType>::which_child(TreeNode* parent, TreeNode* child)
{
	return parent != nullptr && parent->child[1] == child;
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::connect_node(TreeNode* parent, TreeNode* child, bool which)
{
	if (parent == nullptr)
		root = child;
	else
		parent->child[which] = child;
	if (child != nullptr)
		child->parent = parent;
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::update_size(TreeNode* current)
{
	current->size = 1;
	if (current->child[0] != nullptr)
		current->size += current->child[0]->size;
	if (current->child[1] != nullptr)
		current->size += current->child[1]->size;
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::rotate_node(TreeNode* current)
{
	TreeNode* f = current->parent, * g = f->parent;
	bool k = which_child(f, current);
	connect_node(f, current->child[!k], k);
	connect_node(g, current, which_child(g, f));
	connect_node(current, f, !k);
	update_size(f);
	update_size(current);
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::make_splay(TreeNode* current, TreeNode* target)
{
	while (target != current->parent)
	{
		TreeNode* f = current->parent, * g = f->parent;
		if (g == target)
			rotate_node(current);
		else
		{
			if (which_child(g, f) ^ which_child(f, current))
				rotate_node(current), rotate_node(current);
			else
				rotate_node(f), rotate_node(current);
		}
	}
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::delete_node(int rank)
{
	if (root == nullptr)
		return;
	if (root->size == 1)
	{
		delete root;
		root = nullptr;
		return;
	}
	if (rank == 1)
	{
		make_splay(rank_search(root, rank), nullptr);
		delete root;
		root = root->child[1];
		root->parent = nullptr;
	}
	else if (rank == root->size)
	{
		make_splay(rank_search(root, rank), nullptr);
		delete root;
		root = root->child[0];
		root->parent = nullptr;
	}
	else
	{
		make_splay(rank_search(root, rank - 1), nullptr);
		make_splay(rank_search(root, rank + 1), root);
		delete root->child[1]->child[0];
		root->child[1]->child[0] = nullptr;
		update_size(root->child[1]);
		update_size(root);
	}
}

template<typename KeyType, typename DataType>
inline typename SplayTree<KeyType, DataType>::TreeNode* SplayTree<KeyType, DataType>::rank_search(TreeNode* current, int rank)
{
	int remain = 0;
	if (current->child[0] != nullptr)
		remain += current->child[0]->size;
	if (rank == remain + 1)
		return current;
	if (rank <= remain)
		return rank_search(current->child[0], rank);
	else
		return rank_search(current->child[1], rank - remain - 1);
}

template<typename KeyType, typename DataType>
inline typename SplayTree<KeyType, DataType>::TreeNode* SplayTree<KeyType, DataType>::lower_bound(const KeyType& key)
{
	if (root == nullptr)
		return nullptr;
	TreeNode* result = nullptr;
	for (TreeNode* current = root; current != nullptr; current = current->child[key > current->data.key])
		if (current->data.key >= key)
			result = current;
	return result;
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::tree_traverse(TreeNode* current) const
{
	if (current == nullptr)
		return;
	std::cout << (current->data).key << "\t" << (current->data).data << "\t" << current->size << "\t";
	if (current->child[0] == nullptr)
		std::cout << "L: *\t";
	else
		std::cout << "L: " << (current->child[0]->data).key << "\t";
	if (current->child[1] == nullptr)
		std::cout << "R: *\t";
	else
		std::cout << "R: " << (current->child[1]->data).key << "\t";
	std::cout << "\n";
	tree_traverse(current->child[0]);
	tree_traverse(current->child[1]);
}

template<typename KeyType, typename DataType>
inline SplayTree<KeyType, DataType>::SplayTree()
{
	root = nullptr;
}

template<typename KeyType, typename DataType>
inline SplayTree<KeyType, DataType>::~SplayTree()
{
	make_empty(root);
	root = nullptr;
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::clear()
{
	make_empty(root);
	root = nullptr;
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	if (root == nullptr)
	{
		root = new TreeNode(element, 1);
		return;
	}
	for (TreeNode* current = root; current != nullptr; current = current->child[element.key > current->data.key])
	{
		bool which = element.key > current->data.key;
		if (current->child[which] == nullptr)
		{
			current->child[which] = new TreeNode(element, 1, current);
			make_splay(current->child[which], nullptr);
			break;
		}
	}
}

template<typename KeyType, typename DataType>
inline void SplayTree<KeyType, DataType>::remove(const KeyType& key)
{
	TreeNode* target = lower_bound(key);
	if (target != nullptr && target->data.key == key)
	{
		make_splay(target, nullptr);
		int rank = 0;
		if (root->child[0] != nullptr)
			rank = root->child[0]->size;
		delete_node(rank + 1);
	}
}

template<typename KeyType, typename DataType>
inline int SplayTree<KeyType, DataType>::rank(const KeyType& key)
{
	TreeNode* target = lower_bound(key);
	if (target == nullptr)
		return target->size + 1;
	make_splay(target, nullptr);
	int result = 1;
	if (target->child[0] != nullptr)
		result += target->child[0]->size;
	return result;
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* SplayTree<KeyType, DataType>::find(const KeyType& key) const
{
	TreeNode* current = root;
	while (current != nullptr && (current->data).key != key)
	{
		if ((current->data).key > key)
			current = current->child[0];
		else
			current = current->child[1];
	}
	if (current == nullptr)
		return nullptr;
	else
		return &(current->data);
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* SplayTree<KeyType, DataType>::fetch(int rank)
{
	return &(rank_search(root, rank)->data);
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* SplayTree<KeyType, DataType>::precursor(const KeyType& key)
{
	SetElement<KeyType, DataType>* result = nullptr;
	for (TreeNode* current = root; current != nullptr; current = current->child[key > current->data.key])
		if (current->data.key < key)
			if (result == nullptr || current->data.key > result->key)
				result = &(current->data);
	return result;
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* SplayTree<KeyType, DataType>::successor(const KeyType& key)
{
	SetElement<KeyType, DataType>* result = nullptr;
	for (TreeNode* current = root; current != nullptr; current = current->child[key >= current->data.key])
		if (current->data.key > key)
			if (result == nullptr || current->data.key < result->key)
				result = &(current->data);
	return result;
}