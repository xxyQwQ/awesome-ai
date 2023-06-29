#pragma once
#include <iostream>
#include "Exception.h"
#include "LinkedStack.h"
#include "DynamicSet.h"

template <typename KeyType, typename DataType>
class RBTree : public DynamicSet<KeyType, DataType>
{
private:
	enum class NodeType
	{
		RED,
		BLACK
	};
	struct TreeNode
	{
		SetElement<KeyType, DataType> data;
		TreeNode* parent, * left, * right;
		NodeType color;
		TreeNode() :data(), parent(nullptr), left(nullptr), right(nullptr), color(NodeType::BLACK) {}
		TreeNode(const SetElement<KeyType, DataType>& _data, TreeNode* _parent = nullptr, TreeNode* _left = nullptr, TreeNode* _right = nullptr, NodeType _color = NodeType::BLACK) : data(_data), parent(_parent), left(_left), right(_right), color(_color) {}
		TreeNode& operator=(TreeNode& another)
		{
			data = another.data, parent = another.parent, left = another.left, right = another.right, color = another.color;
			return *this;
		}
	};
	TreeNode* root, * empty;
	int count;
	TreeNode* tree_maximum(TreeNode* target) const;
	TreeNode* tree_minimum(TreeNode* target) const;
	TreeNode* place_specific(const KeyType& key) const;
	TreeNode* place_precursor(TreeNode* target) const;
	TreeNode* place_successor(TreeNode* target) const;
	void left_rotate(TreeNode* current);
	void right_rotate(TreeNode* current);
	void tree_replace(TreeNode* current, TreeNode* another);
	void insert_repair(TreeNode* current);
	void remove_repair(TreeNode* current);
	void make_empty(TreeNode* current);
public:
	RBTree();
	~RBTree();
	int size() const;
	const SetElement<KeyType, DataType>* precursor(const KeyType& key) const;
	const SetElement<KeyType, DataType>* successor(const KeyType& key) const;
	const SetElement<KeyType, DataType>* find(const KeyType& key) const;
	void insert(const SetElement<KeyType, DataType>& element);
	void remove(const KeyType& key);
	void traverse(); // preorder
	void clear();
};

template <typename KeyType, typename DataType>
inline typename RBTree<KeyType, DataType>::TreeNode* RBTree<KeyType, DataType>::place_specific(const KeyType& key) const
{
	TreeNode* current = root;
	while (current != empty && (current->data).key != key)
	{
		if ((current->data).key > key)
			current = current->left;
		else
			current = current->right;
	}
	return current;
}

template <typename KeyType, typename DataType>
inline typename RBTree<KeyType, DataType>::TreeNode* RBTree<KeyType, DataType>::tree_maximum(RBTree<KeyType, DataType>::TreeNode* target) const
{
	TreeNode* current = target;
	while (current->right != empty)
		current = current->right;
	return current;
}

template <typename KeyType, typename DataType>
inline typename RBTree<KeyType, DataType>::TreeNode* RBTree<KeyType, DataType>::tree_minimum(RBTree<KeyType, DataType>::TreeNode* target) const
{
	TreeNode* current = target;
	while (current->left != empty)
		current = current->left;
	return current;
}

template <typename KeyType, typename DataType>
inline typename RBTree<KeyType, DataType>::TreeNode* RBTree<KeyType, DataType>::place_precursor(RBTree<KeyType, DataType>::TreeNode* target) const
{
	if (target->left != empty)
		return tree_maximum(target->left);
	TreeNode* current = target->parent;
	while (current != empty && target == current->left)
		target = current, current = current->parent;
	return current;
}

template <typename KeyType, typename DataType>
inline typename RBTree<KeyType, DataType>::TreeNode* RBTree<KeyType, DataType>::place_successor(RBTree<KeyType, DataType>::TreeNode* target) const
{
	if (target->right != empty)
		return tree_minimum(target->right);
	TreeNode* current = target->parent;
	while (current != empty && target == current->right)
		target = current, current = current->parent;
	return current;
}

template <typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::left_rotate(TreeNode* current)
{
	if (current->right == empty)
		return;
	TreeNode* child = current->right;
	current->right = child->left;
	if (child->left != empty)
		child->left->parent = current;
	child->parent = current->parent;
	if (current->parent == empty)
		root = child;
	else if (current == current->parent->left)
		current->parent->left = child;
	else
		current->parent->right = child;
	child->left = current;
	current->parent = child;
}

template <typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::right_rotate(TreeNode* current)
{
	if (current->left == empty)
		return;
	TreeNode* child = current->left;
	current->left = child->right;
	if (child->right != empty)
		child->right->parent = current;
	child->parent = current->parent;
	if (current->parent == empty)
		root = child;
	else if (current == current->parent->left)
		current->parent->left = child;
	else
		current->parent->right = child;
	child->right = current;
	current->parent = child;
}

template <typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::tree_replace(TreeNode* current, TreeNode* another)
{
	if (current->parent == empty)
		root = another;
	else if (current == current->parent->left)
		current->parent->left = another;
	else
		current->parent->right = another;
	another->parent = current->parent;
}

template<typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::insert_repair(TreeNode* current)
{
	while (current->parent->color == NodeType::RED)
	{
		if (current->parent == current->parent->parent->left)
		{
			TreeNode* fellow = current->parent->parent->right;
			if (fellow->color == NodeType::RED)
			{
				current->parent->color = NodeType::BLACK;
				fellow->color = NodeType::BLACK;
				current->parent->parent->color = NodeType::RED;
				current = current->parent->parent;
			}
			else
			{
				if (current == current->parent->right)
				{
					current = current->parent;
					left_rotate(current);
				}
				current->parent->color = NodeType::BLACK;
				current->parent->parent->color = NodeType::RED;
				right_rotate(current->parent->parent);
			}
		}
		else
		{
			TreeNode* fellow = current->parent->parent->left;
			if (fellow->color == NodeType::RED)
			{
				current->parent->color = NodeType::BLACK;
				fellow->color = NodeType::BLACK;
				current->parent->parent->color = NodeType::RED;
				current = current->parent->parent;
			}
			else
			{
				if (current == current->parent->left)
				{
					current = current->parent;
					right_rotate(current);
				}
				current->parent->color = NodeType::BLACK;
				current->parent->parent->color = NodeType::RED;
				left_rotate(current->parent->parent);
			}
		}
	}
	root->color = NodeType::BLACK;
}

template<typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::remove_repair(TreeNode* current)
{
	while (current != root && current->color == NodeType::BLACK)
	{
		if (current == current->parent->left)
		{
			TreeNode* fellow = current->parent->right;
			if (fellow->color == NodeType::RED)
			{
				fellow->color = NodeType::BLACK;
				fellow->parent->color = NodeType::RED;
				right_rotate(fellow);
				fellow = current->parent->right;
			}
			if (fellow->left->color == NodeType::BLACK && fellow->right->color == NodeType::BLACK)
			{
				fellow->color = NodeType::RED;
				current = current->parent;
			}
			else
			{
				if (fellow->right->color == NodeType::BLACK)
				{
					fellow->left->color = NodeType::BLACK;
					fellow->color = NodeType::RED;
					right_rotate(fellow);
					fellow = current->parent->right;
				}
				fellow->color = current->parent->color;
				current->parent->color = NodeType::BLACK;
				fellow->right->color = NodeType::BLACK;
				left_rotate(current->parent);
				current = root;
			}
		}
		else
		{
			TreeNode* fellow = current->parent->left;
			if (fellow->color == NodeType::RED)
			{
				fellow->color = NodeType::BLACK;
				fellow->parent->color = NodeType::RED;
				left_rotate(fellow);
				fellow = current->parent->left;
			}
			if (fellow->left->color == NodeType::BLACK && fellow->right->color == NodeType::BLACK)
			{
				fellow->color = NodeType::RED;
				current = current->parent;
			}
			else
			{
				if (fellow->left->color == NodeType::BLACK)
				{
					fellow->right->color = NodeType::BLACK;
					fellow->color = NodeType::RED;
					left_rotate(fellow);
					fellow = current->parent->left;
				}
				fellow->color = current->parent->color;
				current->parent->color = NodeType::BLACK;
				fellow->left->color = NodeType::BLACK;
				right_rotate(current->parent);
				current = root;
			}
		}
	}
	current->color = NodeType::BLACK;
}

template <typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::make_empty(TreeNode* current)
{
	if (current == empty)
		return;
	make_empty(current->left);
	make_empty(current->right);
	delete current;
}

template<typename KeyType, typename DataType>
inline RBTree<KeyType, DataType>::RBTree()
{
	empty = new TreeNode();
	root = empty;
	count = 0;
}

template<typename KeyType, typename DataType>
inline RBTree<KeyType, DataType>::~RBTree()
{
	make_empty(root);
	delete empty;
}

template<typename KeyType, typename DataType>
inline int RBTree<KeyType, DataType>::size() const
{
	return count;
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* RBTree<KeyType, DataType>::precursor(const KeyType& key) const
{
	TreeNode* target = place_specific(key);
	if (target == empty)
		throw InvalidQuery("Error: No such element in set");
	TreeNode* result = place_precursor(target);
	if (result == empty)
		return nullptr;
	else
		return &(result->data);
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* RBTree<KeyType, DataType>::successor(const KeyType& key) const
{
	TreeNode* target = place_specific(key);
	if (target == empty)
		throw InvalidQuery("Error: No such element in set");
	TreeNode* result = place_successor(target);
	if (result == empty)
		return nullptr;
	else
		return &(result->data);
}

template<typename KeyType, typename DataType>
inline const SetElement<KeyType, DataType>* RBTree<KeyType, DataType>::find(const KeyType& key) const
{
	TreeNode* result = place_specific(key);
	if (result == empty)
		return nullptr;
	else
		return &(result->data);
}

template <typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::insert(const SetElement<KeyType, DataType>& element)
{
	TreeNode* addition = new TreeNode(element, empty, empty, empty, NodeType::RED);
	TreeNode* current = root, * parent = empty;
	while (current != empty)
	{
		parent = current;
		if (element.key < (current->data).key)
			current = current->left;
		else
			current = current->right;
	}
	addition->parent = parent;
	if (parent == empty)
		root = addition;
	else if ((addition->data).key < (parent->data).key)
		parent->left = addition;
	else
		parent->right = addition;
	insert_repair(addition);
	count++;
}

template<typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::remove(const KeyType& key)
{
	TreeNode* current = place_specific(key);
	if (current == empty)
		throw InvalidModify("No such element in set");
	TreeNode* target = current, * temp = empty;
	NodeType color = target->color;
	if (current->left == empty)
	{
		temp = current->right;
		tree_replace(current, current->right);
	}
	else if (current->right == empty)
	{
		temp = current->left;
		tree_replace(current, current->left);
	}
	else
	{
		target = tree_minimum(current->right);
		color = target->color;
		temp = target->right;
		if (target->parent == current)
			temp->parent = target;
		else
		{
			tree_replace(target, target->right);
			target->right = current->right;
			target->right->parent = target;
		}
		tree_replace(current, target);
		target->left = current->left;
		target->left->parent = target;
		target->color = current->color;
	}
	if (color == NodeType::BLACK)
		remove_repair(temp);
	delete current;
	count--;
}

template<typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::traverse()
{
	LinkedStack<TreeNode*> stack;
	TreeNode* current = root;
	std::cout << "Elements in set are as follow:\n";
	while (current != empty || !stack.empty())
	{
		while (current != empty)
		{
			stack.push(current);
			std::cout << (current->data).key << "\t" << (current->data).data << "\t";
			if (current->color == NodeType::BLACK)
				std::cout << "BLACK\t";
			else
				std::cout << "RED\t";
			if (current->left == empty)
				std::cout << "L: *\t";
			else
				std::cout << "L: " << (current->left->data).key << "\t";
			if (current->right == empty)
				std::cout << "R: *\t";
			else
				std::cout << "R: " << (current->right->data).key << "\t";
			std::cout << "\n";
			current = current->left;
		}
		if (!stack.empty())
		{
			current = stack.pop();
			current = current->right;
		}
	}
}

template <typename KeyType, typename DataType>
inline void RBTree<KeyType, DataType>::clear()
{
	make_empty(root);
	root = empty;
	count = 0;
}