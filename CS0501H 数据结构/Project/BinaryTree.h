#pragma once

template <typename ElementType>
class BinaryTree
{
public:
	virtual ~BinaryTree() {}
	virtual void doClear() = 0;
	virtual bool isEmpty() const = 0;
	virtual ElementType getRoot(ElementType flag) const = 0;
	virtual ElementType getParent(ElementType target, ElementType flag) const = 0;
	virtual ElementType getLeftChild(ElementType target, ElementType flag) const = 0;
	virtual ElementType getRightChild(ElementType target, ElementType flag) const = 0;
	virtual void deleteLeftChild(ElementType target) = 0;
	virtual void deleteRightChild(ElementType target) = 0;
	virtual void insertLeftChild(ElementType target, ElementType extra) = 0;
	virtual void insertRightChild(ElementType target, ElementType extra) = 0;
	virtual void doPreOrder() const = 0;
	virtual void doInOrder() const = 0;
	virtual void doPostOrder() const = 0;
	virtual void doLevelOrder() const = 0;
};