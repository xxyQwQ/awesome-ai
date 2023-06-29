#pragma once
#include <cstring>
#include <exception>

class IndexExceed : public std::exception
{
private:
	char* message;
public:
	IndexExceed(const char* content = "Error: Index is out of range")
	{
		message = new char[strlen(content) + 1];
		strcpy(message, content);
	}
	virtual ~IndexExceed()
	{
		delete[] message;
	}
	virtual const char* what() const throw()
	{
		return message;
	}
};

class InvalidQuery : public std::exception
{
private:
	char* message;
public:
	InvalidQuery(const char* content = "Error: Query is invalid")
	{
		message = new char[strlen(content) + 1];
		strcpy(message, content);
	}
	virtual ~InvalidQuery()
	{
		delete[] message;
	}
	virtual const char* what() const throw()
	{
		return message;
	}
};

class InvalidModify : public std::exception
{
private:
	char* message;
public:
	InvalidModify(const char* content = "Error: Modification is invalid")
	{
		message = new char[strlen(content) + 1];
		strcpy(message, content);
	}
	virtual ~InvalidModify()
	{
		delete[] message;
	}
	virtual const char* what() const throw()
	{
		return message;
	}
};

class DivideByZero : public std::exception
{
private:
	char* message;
public:
	DivideByZero(const char* content = "Error: Value is divided by zero")
	{
		message = new char[strlen(content) + 1];
		strcpy(message, content);
	}
	virtual ~DivideByZero()
	{
		delete[] message;
	}
	virtual const char* what() const throw()
	{
		return message;
	}
};

class MissingElement : public std::exception
{
private:
	char* message;
public:
	MissingElement(const char* content = "Error: Necessary element is missing")
	{
		message = new char[strlen(content) + 1];
		strcpy(message, content);
	}
	virtual ~MissingElement()
	{
		delete[] message;
	}
	virtual const char* what() const throw()
	{
		return message;
	}
};

class EmptyContainer : public std::exception
{
private:
	char* message;
public:
	EmptyContainer(const char* content = "Error: The container is already empty")
	{
		message = new char[strlen(content) + 1];
		strcpy(message, content);
	}
	virtual ~EmptyContainer()
	{
		delete[] message;
	}
	virtual const char* what() const throw()
	{
		return message;
	}
};