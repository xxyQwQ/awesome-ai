#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <crtdbg.h>
#include "Exception.h"
#include "Function.h"
#include "SequentialList.h"
#include "SingleLinkedList.h"
#include "DoubleLinkedList.h"
#include "BigInteger.h"
#include "Polynomial.h"
#include "SequentialStack.h"
#include "LinkedStack.h"
#include "ExpressionStack.h"
#include "SequentialQueue.h"
#include "LinkedQueue.h"
#include "SequentialString.h"
#include "LinkedString.h"
#include "PriorityQueue.h"
#include "SimulatorQueue.h"
#include "LinkedBinaryTree.h"
#include "HuffmanTree.h"
#include "ExpressionTree.h"
#include "StaticSet.h"
#include "BinarySearchTree.h"
#include "SplayTree.h"
#include "AVLTree.h"
#include "RBTree.h"
#include "AATree.h"
#include "CloseHashTable.h"
#include "OpenHashTable.h"
#include "FindUnionSet.h"
#include "AdjacencyMatrixGraph.h"
#include "AdjacencyListGraph.h"

void ExceptionTest()
{
	printf("---------- Exception ----------\n");
	try
	{
		DoubleLinkedList<int> L;
		for (int i = 0; i < 10; i++)
			L.push_back(i);
		printf("%d\n", L.fetch(10));
		for (int i = 0; i < 10; i++)
			L.push_front(i + 10);
	}
	catch (const IndexExceed& E)
	{
		printf("%s\n", E.what());
	}
	catch (const InvalidModify& E)
	{
		printf("%s\n", E.what());
	}
	try
	{
		BigInteger x(10), y(0);
		BigInteger z = x / y;
		std::cout << z << std::endl;
	}
	catch (const DivideByZero& E)
	{
		printf("%s\n", E.what());
	}
	try
	{
		SequentialStack<int> S;
		S.pop();
		S.push(1);
		printf("%d\n", S.top());
	}
	catch (const std::exception& E)
	{
		printf("%s\n", E.what());
	}
	try
	{
		SequentialQueue<int> Q;
		Q.pop();
		Q.push(1);
		printf("%d\n", Q.front());
	}
	catch (const std::exception& E)
	{
		printf("%s\n", E.what());
	}
	try
	{
		PriorityQueue<int> Q;
		Q.pop();
		Q.push(1);
		printf("%d\n", Q.top());
	}
	catch (const std::exception& E)
	{
		printf("%s\n", E.what());
	}
	printf("---------- Exception ----------\n\n");
}

void FunctionTest()
{
	printf("---------- Function ----------\n");
	int a[10] = { 37,12,58,4,99,42,15,9,23,74 };
	printf("original array:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	insert_sort(a, 10);
	printf("insert sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	random_shuffle(a, 10);
	printf("random shuffle:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	shell_sort(a, 10);
	printf("shell sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	random_shuffle(a, 10);
	printf("random shuffle:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	select_sort(a, 10);
	printf("select sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	random_shuffle(a, 10);
	printf("random shuffle:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	bubble_sort(a, 10);
	printf("bubble sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	random_shuffle(a, 10);
	printf("random shuffle:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	heap_sort(a, 10);
	printf("heap sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	random_shuffle(a, 10);
	printf("random shuffle:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	quick_sort(a, 10);
	printf("quick sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	random_shuffle(a, 10);
	printf("random shuffle:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	merge_sort(a, 10);
	printf("merge sort:\t");
	for (int i = 0; i < 10; i++)
		printf("%d\t", a[i]);
	printf("\n");
	printf("---------- Function ----------\n\n");
}

void SequentialListTest()
{
	printf("---------- SequentialList ----------\n");
	int a[] = { 1, 2, 3, 5, 8, 13, 21, 34, 55, 89 };
	SequentialList<int> s(5);
	for (int i = 0; i < 10; i++)
		s.append(a[i]);
	s.traverse();
	if (s.find(4) == -1)
		printf("Element 4 was not found in list\n");
	printf("Element 34 was found by index %d\n", s.find(34));
	printf("Element %d was fetched by index 3\n", s.fetch(3));
	printf("Element %d was visited by index 3\n", s[3]);
	for (int i = 1; i <= 4; i++)
		s.remove(3);
	s.insert(3, 114514);
	printf("Current length of list is %d\n", s.length());
	s.traverse();
	printf("---------- SequentialList ----------\n\n");
}

void SingleLinkedListTest()
{
	printf("---------- SingleLinkedList ----------\n");
	int a[] = { 1, 4, 7, 10, 1, 4, 7, 10, 1, 4 };
	SingleLinkedList<int> s;
	for (int i = 0; i < 10; i++)
		s.append(a[i]);
	s.traverse();
	if (s.find(5) == -1)
		printf("Element 5 was not found in list\n");
	printf("Element 10 was found by index %d\n", s.find(10));
	printf("Element %d was fetched by index 3\n", s.fetch(3));
	for (int i = 1; i <= 3; i++)
		s.remove(2);
	s.insert(2, 114514);
	printf("Current length of list is %d\n", s.length());
	s.traverse();
	printf("Erase relevant element by index 1\n");
	s.erase(1);
	s.traverse();
	printf("---------- SingleLinkedList ----------\n\n");
}

void DoubleLinkedListTest()
{
	printf("---------- DoubleLinkedList ----------\n");
	DoubleLinkedList<int> s;
	for (int i = 1; i <= 5; i++)
	{
		s.push_front(2 * i - 1);
		s.push_back(2 * i);
	}
	s.traverse();
	if (s.find(11) == -1)
		printf("Element 11 was not found in list\n");
	printf("Element 8 was found by index %d\n", s.find(8));
	printf("Element %d was fetched by index 4\n", s.fetch(4));
	for (int i = 1; i <= 3; i++)
		s.remove(2);
	s.insert(2, 114514);
	printf("Current length of list is %d\n", s.length());
	s.traverse();
	printf("List was reversed in order\n");
	s.reverse();
	s.traverse();
	printf("Element 8 was found by index %d\n", s.find(8));
	printf("Clear all element in list\n");
	s.clear();
	if (s.find(8) == -1)
		printf("Element 8 was not found in list\n");
	printf("Current length of list is %d\n", s.length());
	printf("---------- DoubleLinkedList ----------\n\n");
}

void BigIntegerTest()
{
	using namespace std;
	printf("---------- BigInteger ----------\n");
	BigInteger a("1"), b, c("-1"), d(2), e(-2);
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	b = c;
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	a += b;
	b += c;
	c += d;
	e = d + d;
	d = b + b;
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	a -= b;
	b -= c;
	d = c - e;
	c = d - (-d);
	e = d - d;
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	a = a * a * a;
	b *= a;
	c *= b;
	d *= -1;
	a *= d;
	e *= c;
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	c /= a;
	b /= c;
	e /= d;
	d /= c;
	b /= c;
	a /= 5;
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	b = c % a;
	d = c % 5;
	e = c % b;
	a = c % -4;
	c = -7 % c;
	b %= 1;
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	a = power(2, 0);
	b = power(2, 1);
	c = power(2, 10);
	d = power(2, 30);
	e = power(2, 100);
	cout << a << "\t" << b << "\t" << c << "\t" << d << "\t" << e << endl;
	printf("---------- BigInteger ----------\n\n");
}

/*
A polynomial input sample:
5 0
-3 1
0 2
-9 4
12 5
-1 -1
-4 0
3 1
1 2
9 4
-3 5
-4 7
0 9
-1 -1
*/

void PolynomialTest()
{
	printf("---------- Polynomial ----------\n");
	Polynomial P, Q;
	P.input();
	P.output();
	Q.input();
	Q.output();
	Polynomial R = P + Q;
	printf("Two polynomial are added to be:\n");
	R.output();
	printf("---------- Polynomial ----------\n\n");
}

void SequentialStackTest()
{
	printf("---------- SequentialStack ----------\n");
	SequentialStack<int> s(5);
	for (int i = 0; i < 5; i++)
		s.push(i);
	s.pop();
	for (int i = 5; i < 10; i++)
		s.push(i);
	while (!s.empty())
	{
		printf("%d ", s.top());
		s.pop();
	}
	printf("\n");
	s.clear();
	for (int i = 10; i < 15; i++)
		s.push(i);
	while (!s.empty())
	{
		printf("%d ", s.top());
		s.pop();
	}
	printf("\n");
	printf("---------- SequentialStack ----------\n\n");
}

void LinkedStackTest()
{
	printf("---------- LinkedStack ----------\n");
	LinkedStack<int> s;
	for (int i = 0; i < 5; i++)
		s.push(i);
	s.pop();
	for (int i = 5; i < 10; i++)
		s.push(i);
	while (!s.empty())
	{
		printf("%d ", s.top());
		s.pop();
	}
	printf("\n");
	s.clear();
	for (int i = 10; i < 15; i++)
		s.push(i);
	while (!s.empty())
	{
		printf("%d ", s.top());
		s.pop();
	}
	printf("\n");
	printf("---------- LinkedStack ----------\n\n");
}

void ExpressionStackTest()
{
	printf("---------- ExpressionStack ----------\n");
	try
	{
		char E[] = "(2^10 - 24) / (5 * 2^7 + 20 * 18 - (2^4 - 4^2))";
		ExpressionStack C(E);
		printf("%s = %d\n", E, C.getResult());
		char F[] = "(1 / 2) + (3^5 * 10))";
		ExpressionStack D(F);
		printf("%s = %d\n", F, D.getResult());
	}
	catch (const std::exception& E)
	{
		printf("%s\n", E.what());
	}
	printf("---------- ExpressionStack ----------\n\n");
}

void SequentialQueueTest()
{
	printf("---------- SequentialQueue ----------\n");
	SequentialQueue<int> Q(5);
	for (int i = 0; i < 5; i++)
		Q.push(i);
	Q.pop();
	for (int i = 5; i < 10; i++)
		Q.push(i);
	while (!Q.empty())
	{
		printf("%d ", Q.front());
		Q.pop();
	}
	printf("\n");
	Q.clear();
	for (int i = 10; i < 15; i++)
		Q.push(i);
	while (!Q.empty())
	{
		printf("%d ", Q.front());
		Q.pop();
	}
	printf("\n");
	printf("---------- SequentialQueue ----------\n\n");
}

void LinkedQueueTest()
{
	printf("---------- LinkedQueue ----------\n");
	SequentialQueue<int> Q;
	for (int i = 0; i < 5; i++)
		Q.push(i);
	Q.pop();
	for (int i = 5; i < 10; i++)
		Q.push(i);
	while (!Q.empty())
	{
		printf("%d ", Q.front());
		Q.pop();
	}
	printf("\n");
	Q.clear();
	for (int i = 10; i < 15; i++)
		Q.push(i);
	while (!Q.empty())
	{
		printf("%d ", Q.front());
		Q.pop();
	}
	printf("\n");
	printf("---------- LinkedQueue ----------\n\n");
}

void SequentialStringTest()
{
	printf("---------- SequentialString ----------\n");
	using namespace std;
	SequentialString x("123456789"), y("abc");
	x += y;
	y = x.slice(7, 4);
	x.remove(2, 3);
	cout << x << "\t" << y << endl;
	cout << x.length() << "\t" << y.length() << endl;
	x = SequentialString("abandon");
	y = SequentialString("absolute");
	if (x <= y && x < y)
		printf("abandon < absolute\n");
	x = SequentialString("abcdefg");
	y = SequentialString("bcd");
	if (x != y)
		printf("Unequal!\n");
	for (int i = 0; i < x.length(); i++)
		printf("%c", x[i]);
	printf("\n");
	printf("bcd is found in abcdefg at index %d\n", x.find(y));
	printf("bcd is found in bcd at index %d\n", y.find(y));
	if (y.find(x) == -1)
		printf("abcdefg is not found in bcd\n");
	printf("---------- SequentialString ----------\n\n");
}

void LinkedStringTest()
{
	printf("---------- LinkedString ----------\n");
	using namespace std;
	LinkedString x("123456789"), y("abc");
	x += y;
	y = x.slice(7, 4);
	x.remove(2, 3);
	cout << x << "\t" << y << endl;
	cout << x.length() << "\t" << y.length() << endl;
	x = LinkedString("abandon");
	y = LinkedString("absolute");
	if (x <= y && x < y)
		printf("abandon < absolute\n");
	x = LinkedString("abcdefg");
	y = LinkedString("abcdefg ");
	if (x != y)
		printf("Unequal!\n");
	printf("%d\n", x.length());
	for (int i = 0; i < x.length(); i++)
		printf("%c", x[i]);
	printf("\n");
	printf("---------- LinkedString ----------\n\n");
}

void PriorityQueueTest()
{
	printf("---------- PriorityQueue ----------\n");
	PriorityQueue<int> heap;
	int array[15] = { 7, 4, 1, 6, 5, 9, 3, 0, 8, 2 };
	for (int i = 0; i < 10; i++)
		heap.push(array[i]);
	for (int i = 0; i < 10; i++)
		printf("%d ", heap.pop());
	printf("\n");
	heap.clear();
	for (int i = 0; i < 10; i++)
		array[i] = i;
	PriorityQueue<int, Greater<int>> queue(array, 10);
	for (int i = 1; i <= 5; i++)
		queue.push(100 + i);
	for (int i = 0; i < 15; i++)
	{
		int x = queue.front();
		queue.pop();
		printf("%d ", x);
	}
	printf("\n");
	heap.clear();
	if (heap.empty())
		printf("Now heap is already empty!\n");
	printf("---------- PriorityQueue ----------\n\n");
}

/*
A simulator input sample:
0 2
2 7
4
1000
*/

void SimulatorQueueTest()
{
	printf("---------- SimulatorQueue ----------\n");
	SimulatorQueue S;
	printf("Average wait time is: %d\n", S.GetAverageTime());
	printf("---------- SimulatorQueue ----------\n\n");
}

/*
A binary tree input sample:
1
2 3
4 5
6 7
-1 8
-1 -1
9 -1
-1 -1
-1 -1
-1 -1
*/

void LinkedBinaryTreeTest()
{
	printf("---------- LinkedBinaryTree ----------\n");
	LinkedBinaryTree<int> T;
	T.createTree(-1);
	T.doPreOrder();
	T.doInOrder();
	T.doPostOrder();
	T.doLevelOrder();
	printTree(T, -1);
	printf("Size of binary tree is %d\n", T.size());
	printf("Height of binary tree is %d\n", T.height());
	T.deleteLeftChild(2);
	T.deleteRightChild(3);
	printTree(T, -1);
	printf("Size of binary tree is %d\n", T.size());
	printf("Height of binary tree is %d\n", T.height());
	T.insertLeftChild(2, 10);
	T.insertLeftChild(10, 11);
	T.insertRightChild(10, 12);
	T.insertRightChild(3, 12);
	T.insertLeftChild(12, 13);
	printTree(T, -1);
	printf("Size of binary tree is %d\n", T.size());
	printf("Height of binary tree is %d\n", T.height());
	printf("---------- LinkedBinaryTree ----------\n\n");
}

void HuffmanTreeTest()
{
	printf("---------- HuffmanTree ----------\n");
	char d[] = "abcde";
	int t[] = { 1, 2, 2, 5, 9 };
	HuffmanTree<char> T(d, t, 5);
	HuffmanTree<char>::HuffmanCode C[5];
	T.getCode(C);
	for (int i = 0; i < 5; i++)
		std::cout << C[i].data << ": " << C[i].code << std::endl;
	printf("Total weight: %d\n", T.getWeight());
	printf("---------- HuffmanTree ----------\n\n");
}

void ExpressionTreeTest()
{
	printf("---------- ExpressionTree ----------\n");
	try
	{
		char E[] = "2 * 3 + (1 * 2 * 3 + 6 * 6) * (2 + 3) / 5 + 2 / 2";
		ExpressionStack C(E);
		printf("%s = %d\n", E, C.getResult());
		char F[] = "(1 + 2 - 3) * 10 / (2 - 2)";
		ExpressionStack D(F);
		printf("%s = %d\n", F, D.getResult());
	}
	catch (const std::exception& E)
	{
		printf("%s\n", E.what());
	}
	printf("---------- ExpressionTree ----------\n\n");
}

void StaticSetTest()
{
	printf("---------- StaticSet ----------\n");
	SetElement<int, int> S[15];
	for (int i = 1; i <= 10; i++)
	{
		S[i].key = i * 10;
		S[i].data = 1 << i;
	}
	int p = SequentialSearch(S, 10, 51);
	if (!p)
		printf("Key 51 is not found in table\n");
	p = SequentialSearch(S, 10, 60);
	printf("Key 60: key = %d, data = %d\n", S[p].key, S[p].data);
	p = BinarySearch(S, 10, 0);
	if (!p)
		printf("Key 0 is not found in table\n");
	p = BinarySearch(S, 10, 90);
	printf("Key 90: key = %d, data = %d\n", S[p].key, S[p].data);
	printf("---------- StaticSet ----------\n\n");
}

void BinarySearchTreeTest()
{
	printf("---------- BinarySearchTree ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	BinarySearchTree<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	printf("---------- BinarySearchTree ----------\n\n");
}

void SplayTreeTest()
{
	printf("---------- SplayTree ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	SplayTree<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	printf("---------- SplayTree ----------\n\n");
}

void AVLTreeTest()
{
	printf("---------- AVLTree ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	AVLTree<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	printf("---------- AVLTree ----------\n\n");
}

void RBTreeTest()
{
	printf("---------- RBTree ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	RBTree<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	T.traverse();
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	T.traverse();
	T.clear();
	for (int i = 0; i < 26; i++)
		T.insert(SetElement<int, SequentialString>(i, SequentialString("")));
	T.traverse();
	p = T.precursor(24);
	if (p != nullptr)
		std::cout << "Precursor of 24 is " << p->key << "\n";
	p = T.successor(7);
	if (p != nullptr)
		std::cout << "Successor of 7 is " << p->key << "\n";
	try
	{
		p = T.successor(25);
		if (p == nullptr)
			std::cout << "25 has no successor\n";
		p = T.successor(26);
		if (p == nullptr)
			std::cout << "26 has no successor\n";
	}
	catch (const std::exception& E)
	{
		printf("%s\n", E.what());
	}
	printf("Now size of set is %d\n", T.size());
	printf("---------- RBTree ----------\n\n");
}

void AATreeTest()
{
	printf("---------- AATree ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	AATree<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	T.traverse();
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	T.traverse();
	printf("---------- AATree ----------\n\n");
}

int HashParse(const int& key)
{
	return key * 2 + 1;
}

void CloseHashTableTest()
{
	printf("---------- CloseHashTable ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	CloseHashTable<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	CloseHashTable<int, SequentialString> S(1007, HashParse);
	for (int i = 0; i < 10; i++)
		S.insert(A[i]);
	if (p = S.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	S.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = S.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = S.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	S.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = S.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = S.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	S.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = S.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	printf("---------- CloseHashTable ----------\n\n");
}

void OpenHashTableTest()
{
	printf("---------- OpenHashTable ----------\n");
	SetElement<int, SequentialString> A[] = {
		{10, "aaa"},
		{8, "bbb"},
		{21, "ccc"},
		{87, "ddd"},
		{56, "eee"},
		{4, "fff"},
		{11, "ggg"},
		{3, "hhh"},
		{22, "iiiii"},
		{7, "jjj"}
	};
	OpenHashTable<int, SequentialString> T;
	SetElement<int, SequentialString> x;
	const SetElement<int, SequentialString>* p;
	for (int i = 0; i < 10; i++)
		T.insert(A[i]);
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	T.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = T.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	T.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = T.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	T.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = T.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	OpenHashTable<int, SequentialString> S(1007, HashParse);
	for (int i = 0; i < 10; i++)
		S.insert(A[i]);
	if (p = S.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	S.remove(56);
	std::cout << "Now remove element by key 56\n";
	if (p = S.find(56))
		std::cout << "By key 56 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 56 find nothing\n";
	if (p = S.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	S.remove(21);
	std::cout << "Now remove element by key 21\n";
	if (p = S.find(21))
		std::cout << "By key 21 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 21 find nothing\n";
	if (p = S.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	x = SetElement<int, SequentialString>(30, "xyz");
	S.insert(x);
	std::cout << "Now insert element (30, xyz)\n";
	if (p = S.find(30))
		std::cout << "By key 30 find (" << p->key << ", " << p->data << ")\n";
	else
		std::cout << "By key 30 find nothing\n";
	printf("---------- OpenHashTable ----------\n\n");
}

void FindUnionSetTest()
{
	printf("---------- FindUnionSet ----------\n");
	FindUnionSet S(5);
	S.merge(1, 2);
	S.merge(3, 4);
	if (S.find(1) == S.find(2))
		printf("1, 2 are together\n");
	if (S.find(3) == S.find(4))
		printf("3, 4 are together\n");
	if (S.find(2) != S.find(4))
		printf("2, 4 are not together\n");
	S.merge(1, 3);
	printf("Now merge 1, 3\n");
	if (S.find(2) == S.find(4))
		printf("2, 4 are together\n");
	printf("---------- FindUnionSet ----------\n\n");
}

void AdjacencyMatrixGraphTest()
{
	printf("---------- AdjacencyMatrixGraph ----------\n");
	AdjacencyMatrixGraph<char, int> G(5, "abcde", -1);
	if (!G.exist('a', 'e'))
		printf("edge <a, e> does not exist\n");
	G.insert('a', 'e', 7);
	G.insert('c', 'd', 7);
	printf("now insert edge <a, e> and <c, d>\n");
	if (G.exist('a', 'e'))
		printf("edge <a, e> exists\n");
	if (!G.exist('d', 'c'))
		printf("edge <d, c> does not exist\n");
	printf("there are %d vertices in graph\n", G.vertex_count());
	printf("there are %d edges in graph\n", G.edge_count());
	G.remove('a', 'e');
	printf("now remove edge <a, e>\n");
	if (!G.exist('a', 'e'))
		printf("edge <a, e> does not exist\n");
	printf("there are %d edges in graph\n", G.edge_count());
	printf("---------- AdjacencyMatrixGraph ----------\n\n");
}

void AdjacencyListGraphTest()
{
	printf("---------- AdjacencyListGraph ----------\n");
	AdjacencyListGraph<char, int> G(5, "abcde");
	if (!G.exist('a', 'e'))
		printf("edge <a, e> does not exist\n");
	G.insert('a', 'e', 7);
	G.insert('c', 'd', 7);
	printf("now insert edge <a, e> and <c, d>\n");
	if (G.exist('a', 'e'))
		printf("edge <a, e> exists\n");
	if (!G.exist('d', 'c'))
		printf("edge <d, c> does not exist\n");
	printf("there are %d vertices in graph\n", G.vertex_count());
	printf("there are %d edges in graph\n", G.edge_count());
	G.remove('a', 'e');
	printf("now remove edge <a, e>\n");
	if (!G.exist('a', 'e'))
		printf("edge <a, e> does not exist\n");
	printf("there are %d edges in graph\n", G.edge_count());
	G.insert('a', 'b', 5);
	G.insert('a', 'c', 2);
	G.insert('c', 'e', 4);
	G.depth_first_search();
	G.breadth_first_search();
	G.topological_sort();
	int v[6] = { 1,2,3,4,5,6 };
	AdjacencyListGraph<int, int> E(6, v);
	E.insert(1, 2, 1);
	E.insert(1, 3, 3);
	E.insert(1, 4, 5);
	E.insert(2, 3, 1);
	E.insert(2, 5, 2);
	E.insert(3, 4, 1);
	E.insert(3, 6, 2);
	E.insert(4, 5, 1);
	E.insert(4, 6, 2);
	E.insert(5, 6, 2);
	E.critical_path();
	printf("---------- AdjacencyListGraph ----------\n\n");
}

void KruskalTest()
{
	printf("---------- Kruskal ----------\n");
	int V[6] = { 1, 2, 3, 4, 5, 6 };
	AdjacencyListGraph<int, int> G(6, V);
	G.insert(1, 2, 6), G.insert(2, 1, 6);
	G.insert(1, 3, 1), G.insert(3, 1, 1);
	G.insert(1, 4, 5), G.insert(4, 1, 5);
	G.insert(2, 3, 5), G.insert(3, 2, 5);
	G.insert(2, 5, 3), G.insert(5, 2, 3);
	G.insert(3, 4, 5), G.insert(4, 3, 5);
	G.insert(3, 5, 6), G.insert(5, 3, 6);
	G.insert(3, 6, 4), G.insert(6, 3, 4);
	G.insert(4, 6, 2), G.insert(6, 4, 2);
	G.insert(5, 6, 6), G.insert(6, 5, 6);
	G.kruskal();
	printf("---------- Kruskal ----------\n\n");
}

void PrimTest()
{
	printf("---------- Prim ----------\n");
	int V[6] = { 1, 2, 3, 4, 5, 6 };
	AdjacencyListGraph<int, int> G(6, V);
	G.insert(1, 2, 6), G.insert(2, 1, 6);
	G.insert(1, 3, 1), G.insert(3, 1, 1);
	G.insert(1, 4, 5), G.insert(4, 1, 5);
	G.insert(2, 3, 5), G.insert(3, 2, 5);
	G.insert(2, 5, 3), G.insert(5, 2, 3);
	G.insert(3, 4, 5), G.insert(4, 3, 5);
	G.insert(3, 5, 6), G.insert(5, 3, 6);
	G.insert(3, 6, 4), G.insert(6, 3, 4);
	G.insert(4, 6, 2), G.insert(6, 4, 2);
	G.insert(5, 6, 6), G.insert(6, 5, 6);
	G.prim();
	printf("---------- Prim ----------\n\n");
}

void UnweightedDistanceTest()
{
	printf("---------- UnweightedDistance ----------\n");
	int V[6] = { 1, 2, 3, 4, 5, 6 };
	AdjacencyListGraph<int, int> G(6, V);
	G.insert(1, 2, 6), G.insert(2, 1, 6);
	G.insert(1, 3, 1), G.insert(3, 1, 1);
	G.insert(1, 4, 5), G.insert(4, 1, 5);
	G.insert(2, 3, 5), G.insert(3, 2, 5);
	G.insert(2, 5, 3), G.insert(5, 2, 3);
	G.insert(3, 4, 5), G.insert(4, 3, 5);
	G.insert(3, 5, 6), G.insert(5, 3, 6);
	G.insert(3, 6, 4), G.insert(6, 3, 4);
	G.insert(4, 6, 2), G.insert(6, 4, 2);
	G.insert(5, 6, 6), G.insert(6, 5, 6);
	G.unweighted_distance(1, 0x3fffffff);
	G.unweighted_distance(3, 0x3fffffff);
	G.unweighted_distance(5, 0x3fffffff);
	printf("---------- UnweightedDistance ----------\n\n");
}

void DijkstraTest()
{
	printf("---------- Dijkstra ----------\n");
	int V[7] = { 0, 1, 2, 3, 4, 5, 6 };
	AdjacencyListGraph<int, int> G(7, V);
	G.insert(0, 1, 2);
	G.insert(0, 3, 1);
	G.insert(1, 3, 3);
	G.insert(1, 4, 10);
	G.insert(2, 0, 4);
	G.insert(2, 5, 5);
	G.insert(3, 2, 2);
	G.insert(3, 4, 2);
	G.insert(3, 5, 8);
	G.insert(3, 6, 4);
	G.insert(4, 6, 6);
	G.insert(6, 5, 1);
	G.dijkstra(1, 0x3fffffff);
	printf("---------- Dijkstra ----------\n\n");
}

void FloydTest()
{
	printf("---------- Floyd ----------\n");
	int V[3] = { 0, 1, 2 };
	AdjacencyMatrixGraph<int, int> G(3, V, 0x3fffffff);
	G.insert(0, 1, 8);
	G.insert(0, 2, 5);
	G.insert(1, 0, 3);
	G.insert(2, 0, 6);
	G.insert(2, 1, 2);
	G.floyd();
	printf("---------- Floyd ----------\n\n");
}

int main()
{
	ExceptionTest();
	FunctionTest();
	SequentialListTest();
	SingleLinkedListTest();
	DoubleLinkedListTest();
	BigIntegerTest();
	SequentialStackTest();
	LinkedStackTest();
	ExpressionStackTest();
	SequentialQueueTest();
	LinkedQueueTest();
	SequentialStringTest();
	LinkedStringTest();
	PriorityQueueTest();
	HuffmanTreeTest();
	ExpressionTreeTest();
	StaticSetTest();
	BinarySearchTreeTest();
	SplayTreeTest();
	AVLTreeTest();
	RBTreeTest();
	AATreeTest();
	CloseHashTableTest();
	OpenHashTableTest();
	FindUnionSetTest();
	AdjacencyMatrixGraphTest();
	AdjacencyListGraphTest();
	KruskalTest();
	PrimTest();
	UnweightedDistanceTest();
	DijkstraTest();
	FloydTest();
	_CrtDumpMemoryLeaks();
	std::cout << "---------- All The Tests Have Been Finished! ----------\n";
	return 0;
}