#pragma once
#include "Exception.h"
#include "Function.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class AVLTree : public DynamicSet<KeyType, DataType>
{
private:
	struct TreeNode
	{
		SetElement<KeyType, DataType> data;
		TreeNode* left, * right;
		int height;
		TreeNode() :data(), left(nullptr), right(nullptr), height(1) {}
		TreeNode(const SetElement<KeyType, DataType>& _data, TreeNode* _left = nullptr, TreeNode* _right = nullptr, int _height = 1) :data(_data), left(_left), right(_right), height(_height) {}
	};
	TreeNode* root;
	void Zig(TreeNode*& current); // LL: right rotate
	void Zag(TreeNode*& current); // RR: left rotate
	void ZagZig(TreeNode*& current); // LR: left rotate, then right rotate
	void ZigZag(TreeNode*& current); // RL: right rotate, then left rotate
	int height(TreeNode* current) const;
	bool adjust(TreeNode*& current, bool type); // type: 0 for left, 1 for right
	void insert(TreeNode*& current, const SetElement<KeyType, DataType>& element);
	bool remove(TreeNode*& current, const KeyType& key);
	void clear(TreeNode* current);
public:
	AVLTree();
	~AVLTree();
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
	void clear();
};

template <typename KeyType, typename DataType>
int AVLTree<KeyType, DataType>::height(TreeNode* current) const
{
	if (current == nullptr)
		return 0;
	else
		return current->height;
}

template <typename KeyType, typename DataType>
bool AVLTree<KeyType, DataType>::adjust(TreeNode*& current, bool type)
{
	if (type)
	{
		if (height(current->left) - height(current->right) == 1)
			return true;
		if (height(current->left) == height(current->right))
		{
			(current->height)--;
			return false;
		}
		if (height(current->left->right) > height(current->left->left))
		{
			ZagZig(current);
			return false;
		}
		Zig(current);
		return height(current->left) != height(current->right);
	}
	else
	{
		if (height(current->right) - height(current->left) == 1)
			return true;
		if (height(current->right) == height(current->left))
		{
			(current->height)--;
			return false;
		}
		if (height(current->right->left) > height(current->right->right))
		{
			ZigZag(current);
			return false;
		}
		Zag(current);
		return height(current->right) != height(current->left);
	}
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::insert(TreeNode*& current, const SetElement<KeyType, DataType>& element)
{
	if (current == nullptr)
		current = new TreeNode(element);
	else if (element.key < (current->data).key)
	{
		insert(current->left, element);
		if (height(current->left) - height(current->right) == 2)
		{
			if (element.key < (current->left->data).key)
				Zig(current);
			else
				ZagZig(current);
		}
	}
	else if (element.key > (current->data).key)
	{
		insert(current->right, element);
		if (height(current->right) - height(current->left) == 2)
		{
			if (element.key > (current->right->data).key)
				Zag(current);
			else
				ZigZag(current);
		}
	}
	current->height = max(height(current->left), height(current->right)) + 1;
}

template <typename KeyType, typename DataType>
bool AVLTree<KeyType, DataType>::remove(TreeNode*& current, const KeyType& key)
{
	if (current == nullptr)
		return true;
	if (key == (current->data).key)
	{
		if (current->left == nullptr || current->right == nullptr)
		{
			TreeNode* temp = current;
			if (current->left != nullptr)
				current = current->left;
			else
				current = current->right;
			delete temp;
			return false;
		}
		else
		{
			TreeNode* temp = current->right;
			while (temp->left != nullptr)
				temp = temp->left;
			current->data = temp->data;
			if (remove(current->right, (temp->data).key))
				return true;
			return adjust(current, 1);
		}
	}
	if (key < (current->data).key)
	{
		if (remove(current->left, key))
			return true;
		return adjust(current, 0);
	}
	else
	{
		if (remove(current->right, key))
			return true;
		return adjust(current, 1);
	}
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::clear(TreeNode* current)
{
	if (current == nullptr)
		return;
	clear(current->left);
	clear(current->right);
	delete current;
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::Zig(TreeNode*& current)
{
	TreeNode* child = current->left;
	current->left = child->right;
	child->right = current;
	current->height = max(height(current->left), height(current->right)) + 1;
	child->height = max(height(child->left), height(child->right)) + 1;
	current = child;
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::Zag(TreeNode*& current)
{
	TreeNode* child = current->right;
	current->right = child->left;
	child->left = current;
	current->height = max(height(current->left), height(current->right)) + 1;
	child->height = max(height(child->left), height(child->right)) + 1;
	current = child;
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::ZagZig(TreeNode*& current)
{
	Zag(current->left);
	Zig(current);
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::ZigZag(TreeNode*& current)
{
	Zig(current->right);
	Zag(current);
}

template <typename KeyType, typename DataType>
const SetElement<KeyType, DataType>* AVLTree<KeyType, DataType>::find(const KeyType& key) const
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

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	insert(root, element);
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::remove(const KeyType& key)
{
	remove(root, key);
}

template <typename KeyType, typename DataType>
AVLTree<KeyType, DataType>::AVLTree()
{
	root = nullptr;
}

template <typename KeyType, typename DataType>
AVLTree<KeyType, DataType>::~AVLTree()
{
	clear(root);
	root = nullptr;
}

template <typename KeyType, typename DataType>
void AVLTree<KeyType, DataType>::clear()
{
	clear(root);
	root = nullptr;
}