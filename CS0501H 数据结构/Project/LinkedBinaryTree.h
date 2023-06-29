#pragma once
#include <iostream>
#include "Exception.h"
#include "LinkedQueue.h"
#include "BinaryTree.h"

template <typename>
class LinkedBinaryTree;

template <typename ElementType>
void printTree(const LinkedBinaryTree<ElementType>& object, ElementType flag);

template <typename ElementType>
class LinkedBinaryTree : public BinaryTree<ElementType>
{
	friend void printTree<ElementType>(const LinkedBinaryTree<ElementType>& object, ElementType flag);

private:
	struct BinaryTreeNode
	{
		BinaryTreeNode* left, * right;
		ElementType data;
		BinaryTreeNode() : left(nullptr), right(nullptr) {}
		BinaryTreeNode(ElementType _data, BinaryTreeNode* _left = nullptr, BinaryTreeNode* _right = nullptr) : data(_data), left(_left), right(_right) {}
		~BinaryTreeNode() {}
	};
	BinaryTreeNode* root;
	BinaryTreeNode* find(ElementType target, BinaryTreeNode* current) const;
	BinaryTreeNode* trace(ElementType target, BinaryTreeNode* current) const;
	void clear(BinaryTreeNode*& current);
	void preOrder(BinaryTreeNode* current) const;
	void inOrder(BinaryTreeNode* current) const;
	void postOrder(BinaryTreeNode* current) const;
	int size(BinaryTreeNode* current) const;
	int height(BinaryTreeNode* current) const;

public:
	LinkedBinaryTree();
	LinkedBinaryTree(ElementType _initial);
	~LinkedBinaryTree();
	void doClear();
	bool isEmpty() const;
	ElementType getRoot(ElementType flag) const;
	ElementType getParent(ElementType target, ElementType flag) const;
	ElementType getLeftChild(ElementType target, ElementType flag) const;
	ElementType getRightChild(ElementType target, ElementType flag) const;
	int size() const;
	int height() const;
	void deleteLeftChild(ElementType target);
	void deleteRightChild(ElementType target);
	void insertLeftChild(ElementType target, ElementType extra);
	void insertRightChild(ElementType target, ElementType extra);
	void doPreOrder() const;
	void doInOrder() const;
	void doPostOrder() const;
	void doLevelOrder() const;
	void createTree(ElementType flag);
};

template <typename ElementType>
typename LinkedBinaryTree<ElementType>::BinaryTreeNode* LinkedBinaryTree<ElementType>::find(ElementType target, LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr)
		return nullptr;
	if (current->data == target)
		return current;
	BinaryTreeNode* temp = find(target, current->left);
	if (temp != nullptr)
		return temp;
	else
		return find(target, current->right);
}

template <typename ElementType>
typename LinkedBinaryTree<ElementType>::BinaryTreeNode* LinkedBinaryTree<ElementType>::trace(ElementType target, LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr || (current->left == nullptr && current->right == nullptr))
		return nullptr;
	if ((current->left != nullptr && current->left->data == target) || (current->right != nullptr && current->right->data == target))
		return current;
	if (current->left != nullptr)
	{
		BinaryTreeNode* temp = trace(target, current->left);
		if (temp != nullptr)
			return temp;
	}
	if (current->right != nullptr)
	{
		BinaryTreeNode* temp = trace(target, current->right);
		if (temp != nullptr)
			return temp;
	}
	return nullptr;
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::clear(BinaryTreeNode*& current)
{
	if (current == nullptr)
		return;
	clear(current->left);
	clear(current->right);
	delete current;
	current = nullptr;
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::preOrder(LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr)
		return;
	std::cout << current->data << " ";
	preOrder(current->left);
	preOrder(current->right);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::inOrder(LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr)
		return;
	inOrder(current->left);
	std::cout << current->data << " ";
	inOrder(current->right);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::postOrder(LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr)
		return;
	postOrder(current->left);
	postOrder(current->right);
	std::cout << current->data << " ";
}

template <typename ElementType>
int LinkedBinaryTree<ElementType>::size(LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr)
		return 0;
	return 1 + size(current->left) + size(current->right);
}

template <typename ElementType>
int LinkedBinaryTree<ElementType>::height(LinkedBinaryTree<ElementType>::BinaryTreeNode* current) const
{
	if (current == nullptr)
		return 0;
	int _left = height(current->left), _right = height(current->right);
	if (_left >= _right)
		return _left + 1;
	else
		return _right + 1;
}

template <typename ElementType>
LinkedBinaryTree<ElementType>::LinkedBinaryTree()
{
	root = nullptr;
}

template <typename ElementType>
LinkedBinaryTree<ElementType>::LinkedBinaryTree(ElementType _initial)
{
	root = new BinaryTreeNode(_initial);
}

template <typename ElementType>
LinkedBinaryTree<ElementType>::~LinkedBinaryTree()
{
	clear(root);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::doClear()
{
	clear(root);
}

template <typename ElementType>
bool LinkedBinaryTree<ElementType>::isEmpty() const
{
	return root == nullptr;
}

template <typename ElementType>
ElementType LinkedBinaryTree<ElementType>::getRoot(ElementType flag) const
{
	if (root == nullptr)
		return flag;
	return root->data;
}

template <typename ElementType>
ElementType LinkedBinaryTree<ElementType>::getParent(ElementType target, ElementType flag) const
{
	if (root == nullptr || root->data == target)
		return flag;
	BinaryTreeNode* temp = trace(target, root);
	if (temp == nullptr)
		return flag;
	return temp->data;
}

template <typename ElementType>
ElementType LinkedBinaryTree<ElementType>::getLeftChild(ElementType target, ElementType flag) const
{
	BinaryTreeNode* temp = find(target, root);
	if (temp == nullptr || temp->left == nullptr)
		return flag;
	return temp->left->data;
}

template <typename ElementType>
ElementType LinkedBinaryTree<ElementType>::getRightChild(ElementType target, ElementType flag) const
{
	BinaryTreeNode* temp = find(target, root);
	if (temp == nullptr || temp->right == nullptr)
		return flag;
	return temp->right->data;
}

template <typename ElementType>
int LinkedBinaryTree<ElementType>::size() const
{
	return size(root);
}

template <typename ElementType>
int LinkedBinaryTree<ElementType>::height() const
{
	return height(root);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::doPreOrder() const
{
	std::cout << "PreOrder: ";
	preOrder(root);
	std::cout << "\n";
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::doInOrder() const
{
	std::cout << "InOrder: ";
	inOrder(root);
	std::cout << "\n";
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::doPostOrder() const
{
	std::cout << "PostOrder: ";
	postOrder(root);
	std::cout << "\n";
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::doLevelOrder() const
{
	std::cout << "LevelOrder: ";
	LinkedQueue<BinaryTreeNode*> queue;
	queue.push(root);
	while (!queue.empty())
	{
		BinaryTreeNode* temp = queue.pop();
		std::cout << temp->data << " ";
		if (temp->left != nullptr)
			queue.push(temp->left);
		if (temp->right != nullptr)
			queue.push(temp->right);
	}
	std::cout << "\n";
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::deleteLeftChild(ElementType target)
{
	BinaryTreeNode* temp = find(target, root);
	if (temp == nullptr)
		return;
	clear(temp->left);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::deleteRightChild(ElementType target)
{
	BinaryTreeNode* temp = find(target, root);
	if (temp == nullptr)
		return;
	clear(temp->right);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::insertLeftChild(ElementType target, ElementType extra)
{
	BinaryTreeNode* temp = find(target, root);
	if (temp == nullptr)
		return;
	if (temp->left == nullptr)
		temp->left = new BinaryTreeNode(extra);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::insertRightChild(ElementType target, ElementType extra)
{
	BinaryTreeNode* temp = find(target, root);
	if (temp == nullptr)
		return;
	if (temp->right == nullptr)
		temp->right = new BinaryTreeNode(extra);
}

template <typename ElementType>
void LinkedBinaryTree<ElementType>::createTree(ElementType flag)
{
	LinkedQueue<BinaryTreeNode*> queue;
	ElementType _root, _left, _right;
	std::cout << "Input root: ";
	std::cin >> _root;
	root = new BinaryTreeNode(_root);
	queue.push(root);
	while (!queue.empty())
	{
		BinaryTreeNode* temp = queue.pop();
		std::cout << "Input two children of " << temp->data << ": ";
		std::cin >> _left >> _right;
		if (_left != flag)
		{
			temp->left = new BinaryTreeNode(_left);
			queue.push(temp->left);
		}
		if (_right != flag)
		{
			temp->right = new BinaryTreeNode(_right);
			queue.push(temp->right);
		}
	}
	std::cout << "Binary tree is created!\n";
}

template <typename ElementType>
void printTree(const LinkedBinaryTree<ElementType>& object, ElementType flag)
{
	LinkedQueue<ElementType> queue;
	queue.push(object.root->data);
	std::cout << "Binary tree is as follow:\n";
	while (!queue.empty())
	{
		ElementType _root, _left, _right;
		_root = queue.pop();
		_left = object.getLeftChild(_root, flag);
		_right = object.getRightChild(_root, flag);
		std::cout << "<" << _root << ">\t" << _left << "\t" << _right << std::endl;
		if (_left != flag)
			queue.push(_left);
		if (_right != flag)
			queue.push(_right);
	}
	std::cout << "Binary tree is printed!\n";
}