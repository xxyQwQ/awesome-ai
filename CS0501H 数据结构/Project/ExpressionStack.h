#pragma once
#include <iostream>
#include <cstring>
#include "Exception.h"
#include "LinkedStack.h"

int power(int a, int k)
{
	int r = 1;
	while (k)
	{
		if (k & 1)
			r *= a;
		a *= a;
		k >>= 1;
	}
	return r;
}

class ExpressionStack
{
private:
	enum class operation
	{
		OPA, // (
		ADD, // +
		SUB, // -
		MUL, // *
		DIV, // /
		EXP, // ^
		CPA, // )
		VAL, // value
		EOL  // flag
	};
	char* expression, * current;
	operation getOperation(int& value);
	void doOperation(operation type, LinkedStack<int>& stack);

public:
	ExpressionStack(char* sequence);
	~ExpressionStack();
	int getResult();
};

ExpressionStack::operation ExpressionStack::getOperation(int& value)
{
	while (*current && *current == ' ')
		current++;
	if (!*current)
		return operation::EOL;
	if ('0' <= *current && *current <= '9')
	{
		value = 0;
		while ('0' <= *current && *current <= '9')
		{
			value = value * 10 + *current - '0';
			current++;
		}
		return operation::VAL;
	}
	switch (*current)
	{
	case '(':
		current++;
		return operation::OPA;
	case ')':
		current++;
		return operation::CPA;
	case '+':
		current++;
		return operation::ADD;
	case '-':
		current++;
		return operation::SUB;
	case '*':
		current++;
		return operation::MUL;
	case '/':
		current++;
		return operation::DIV;
	case '^':
		current++;
		return operation::EXP;
	}
	return operation::EOL;
}

void ExpressionStack::doOperation(operation type, LinkedStack<int>& stack)
{
	int x, y;
	if (stack.empty())
		throw MissingElement("Error: Right operand is missing");
	else
		y = stack.pop();
	if (stack.empty())
		throw MissingElement("Error: Left operand is missing");
	else
		x = stack.pop();
	switch (type)
	{
	case operation::ADD:
		stack.push(x + y);
		break;
	case operation::SUB:
		stack.push(x - y);
		break;
	case operation::MUL:
		stack.push(x * y);
		break;
	case operation::DIV:
		if (y == 0)
			throw DivideByZero();
		else
			stack.push(x / y);
		break;
	case operation::EXP:
		stack.push(power(x, y));
		break;
	}
}

ExpressionStack::ExpressionStack(char* sequence)
{
	current = expression = new char[strlen(sequence) + 1];
	strcpy(expression, sequence);
}

ExpressionStack::~ExpressionStack()
{
	delete[] expression;
}

int ExpressionStack::getResult()
{
	LinkedStack<operation> OperatorStack;
	LinkedStack<int> OperandStack;
	operation last, top = operation::VAL;
	int value;
	while ((last = getOperation(value)) != operation::EOL)
	{
		switch (last)
		{
		case operation::VAL:
			OperandStack.push(value);
			break;
		case operation::OPA:
			OperatorStack.push(operation::OPA);
			break;
		case operation::CPA:
			while (!OperatorStack.empty() && (top = OperatorStack.pop()) != operation::OPA)
				doOperation(top, OperandStack);
			if (top != operation::OPA)
				throw MissingElement("Error: Left operand is missing");
			break;
		case operation::EXP:
			OperatorStack.push(operation::EXP);
			break;
		case operation::MUL:
		case operation::DIV:
			while (!OperatorStack.empty() && OperatorStack.top() >= operation::MUL)
				doOperation(OperatorStack.pop(), OperandStack);
			OperatorStack.push(last);
			break;
		case operation::ADD:
		case operation::SUB:
			while (!OperatorStack.empty() && OperatorStack.top() != operation::OPA)
				doOperation(OperatorStack.pop(), OperandStack);
			OperatorStack.push(last);
			break;
		}
	}
	while (!OperatorStack.empty())
		doOperation(OperatorStack.pop(), OperandStack);
	if (OperandStack.empty())
		throw MissingElement("Error: There are too few operands");
	int result = OperandStack.pop();
	if (!OperandStack.empty())
		throw MissingElement("Error: There are too few operators");
	return result;
}