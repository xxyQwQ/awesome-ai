#pragma once
#include "Exception.h"

class ExpressionTree
{
private:
	enum class Operation
	{
		VAL, // value
		ADD, // +
		SUB, // -
		MUL, // *
		DIV, // /
		OPA, // (
		CPA, // )
		EOL  // flag
	};
	struct Node
	{
		Operation type;
		int data;
		Node* left, * right;
		Node() : type(Operation::VAL), data(0), left(nullptr), right(nullptr) {}
		Node(Operation _type, int _data = 0, Node* _left = nullptr, Node* _right = nullptr) : type(_type), data(_data), left(_left), right(_right) {}
	};
	Node* root;
	Operation getOperation(char*& content, int& value);
	Node* createTree(char*& expression);
	int getResult(Node* current);
	void clearTree(Node* current);
public:
	ExpressionTree(char* expression);
	~ExpressionTree();
	int getResult();
};

ExpressionTree::Operation ExpressionTree::getOperation(char*& current, int& value)
{
	while (*current == ' ')
		current++;
	if (!*current)
		return Operation::EOL;
	if (0 <= *current && *current <= 9)
	{
		value = 0;
		while (0 <= *current && *current <= 9)
		{
			value = value * 10 + *current - '0';
			current++;
		}
		return Operation::VAL;
	}
	char target = *current;
	current++;
	switch (target)
	{
	case '+':
		return Operation::ADD;
	case '-':
		return Operation::SUB;
	case '*':
		return Operation::MUL;
	case '/':
		return Operation::DIV;
	case '(':
		return Operation::OPA;
	case ')':
		return Operation::CPA;
	default:
		return Operation::EOL;
	}
}

ExpressionTree::Node* ExpressionTree::createTree(char*& expression)
{
	Node* p = nullptr, * r = nullptr;
	Operation type;
	int value;
	while (*expression)
	{
		type = getOperation(expression, value);
		switch (type)
		{
		case Operation::VAL:
		case Operation::OPA:
			if (type == Operation::VAL)
				p = new Node(Operation::VAL, value);
			else
				p = createTree(expression);
			if (r != nullptr)
			{
				if (r->right == nullptr)
					r->right = p;
				else
					r->right->right = p;
			}
			break;
		case Operation::CPA:
		case Operation::EOL:
			return r;
			break;
		case Operation::ADD:
		case Operation::SUB:
			if (r == nullptr)
				r = new Node(type, 0, p);
			else
				r = new Node(type, 0, r);
			break;
		case Operation::MUL:
		case Operation::DIV:
			if (r == nullptr)
				r = new Node(type, 0, p);
			else if (r->type == Operation::MUL || r->type == Operation::DIV)
				r = new Node(type, 0, r);
			else
				r->right = new Node(type, 0, r->right);
			break;
		}
	}
	return r;
}

int ExpressionTree::getResult(Node* current)
{
	if (current->type == Operation::VAL)
		return current->data;
	int left = getResult(current->left);
	int right = getResult(current->right);
	switch (current->type)
	{
	case Operation::ADD:
		current->data = left + right;
		break;
	case Operation::SUB:
		current->data = left - right;
		break;
	case Operation::MUL:
		current->data = left * right;
		break;
	case Operation::DIV:
		if (right == 0)
			throw DivideByZero();
		else
			current->data = left + right;
		break;
	}
	return current->data;
}

void ExpressionTree::clearTree(Node* current)
{
	if (current == nullptr)
		return;
	clearTree(current->left);
	clearTree(current->right);
	delete current;
}

ExpressionTree::ExpressionTree(char* expression)
{
	root = createTree(expression);
}

ExpressionTree::~ExpressionTree()
{
	clearTree(root);
}

int ExpressionTree::getResult()
{
	if (root == nullptr)
		return 0;
	return getResult(root);
}