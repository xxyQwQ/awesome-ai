#pragma once
#include <iostream>
#include "Exception.h"
#include "Function.h"
#include "Graph.h"
#include "LinkedQueue.h"
#include "PriorityQueue.h"
#include "FindUnionSet.h"

template <typename VertexType, typename EdgeType>
class AdjacencyListGraph : public Graph<VertexType, EdgeType>
{
private:
	struct EdgeInfo
	{
		int begin, end;
		EdgeType weight;
		EdgeInfo() : begin(0), end(0), weight() {}
		EdgeInfo(int _begin, int _end, EdgeType _weight) : begin(_begin), end(_end), weight(_weight) {}
		bool operator==(const EdgeInfo& another) const
		{
			return weight == another.weight;
		}
		bool operator<(const EdgeInfo& another) const
		{
			return weight < another.weight;
		}
		bool operator>(const EdgeInfo& another) const
		{
			return weight > another.weight;
		}
	};
	struct EdgeNode
	{
		int target;
		EdgeType weight;
		EdgeNode* next;
		EdgeNode(int _target, EdgeType _weight, EdgeNode* _next = nullptr) : target(_target), weight(_weight), next(_next) {}
	};
	struct VertexNode
	{
		VertexType value;
		EdgeNode* head;
		VertexNode(EdgeNode* _head = nullptr) : value(0), head(_head) {}
		VertexNode(VertexType _value, EdgeNode* _head = nullptr) : value(_value), head(_head) {}
	};
	VertexNode* vertex;
	int position(VertexType x) const;
	void depth_first_search(int current, bool visit[]) const;
	void print_path(int begin, int end, int precursor[]) const;
public:
	AdjacencyListGraph(int number_of_vertex, const VertexType origin_vertex_list[]);
	~AdjacencyListGraph();
	void insert(VertexType x, VertexType y, EdgeType z);
	void remove(VertexType x, VertexType y);
	bool exist(VertexType x, VertexType y) const;
	void depth_first_search() const;
	void breadth_first_search() const;
	void topological_sort() const;
	void critical_path() const;
	void kruskal() const;
	void prim() const;
	void unweighted_distance(VertexType start, EdgeType infinity) const;
	void dijkstra(VertexType start, EdgeType infinity) const;
};

template<typename VertexType, typename EdgeType>
inline int AdjacencyListGraph<VertexType, EdgeType>::position(VertexType x) const
{
	for (int i = 0; i < this->total_vertex; i++)
		if (vertex[i].value == x)
			return i;
	return -1;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::depth_first_search(int current, bool visit[]) const
{
	EdgeNode* p = vertex[current].head;
	std::cout << vertex[current].value << " ";
	visit[current] = true;
	while (p != nullptr)
	{
		if (!visit[p->target])
			depth_first_search(p->target, visit);
		p = p->next;
	}
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::print_path(int begin, int end, int precursor[]) const
{
	if (begin == end)
	{
		std::cout << vertex[begin].value;
		return;
	}
	print_path(begin, precursor[end], precursor);
	std::cout << " -> " << vertex[end].value;
}

template<typename VertexType, typename EdgeType>
inline AdjacencyListGraph<VertexType, EdgeType>::AdjacencyListGraph(int number_of_vertex, const VertexType origin_vertex_list[])
{
	this->total_vertex = number_of_vertex;
	this->total_edge = 0;
	vertex = new VertexNode[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
		vertex[i].value = origin_vertex_list[i];
}

template<typename VertexType, typename EdgeType>
inline AdjacencyListGraph<VertexType, EdgeType>::~AdjacencyListGraph()
{
	EdgeNode* p;
	for (int i = 0; i < this->total_vertex; i++)
		while ((p = vertex[i].head) != nullptr)
		{
			vertex[i].head = p->next;
			delete p;
		}
	delete[] vertex;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::insert(VertexType x, VertexType y, EdgeType z)
{
	int u = position(x), v = position(y);
	vertex[u].head = new EdgeNode(v, z, vertex[u].head);
	this->total_edge++;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::remove(VertexType x, VertexType y)
{
	int u = position(x), v = position(y);
	EdgeNode* p = vertex[u].head;
	if (p == nullptr)
		return;
	if (p->target == v)
	{
		vertex[u].head = p->next;
		delete p;
		this->total_edge--;
		return;
	}
	while (p->next != nullptr && p->next->target != v)
		p = p->next;
	if (p->next != nullptr)
	{
		EdgeNode* q = p->next;
		p->next = q->next;
		delete q;
		this->total_edge--;
	}
}

template<typename VertexType, typename EdgeType>
inline bool AdjacencyListGraph<VertexType, EdgeType>::exist(VertexType x, VertexType y) const
{
	int u = position(x), v = position(y);
	EdgeNode* p = vertex[u].head;
	while (p != nullptr && p->target != v)
		p = p->next;
	return p != nullptr;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::depth_first_search() const
{
	bool* visit = new bool[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
		visit[i] = false;
	std::cout << "Depth first search:\n";
	for (int i = 0; i < this->total_vertex; i++)
	{
		if (visit[i])
			continue;
		depth_first_search(i, visit);
		std::cout << "\n";
	}
	delete[] visit;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::breadth_first_search() const
{
	bool* visit = new bool[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
		visit[i] = false;
	LinkedQueue<int> queue;
	std::cout << "Breadth first search:\n";
	for (int i = 0; i < this->total_vertex; i++)
	{
		if (visit[i])
			continue;
		queue.push(i);
		while (!queue.empty())
		{
			int current = queue.pop();
			if (visit[current])
				continue;
			std::cout << vertex[current].value << " ";
			visit[current] = true;
			EdgeNode* p = vertex[current].head;
			while (p != nullptr)
			{
				if (!visit[p->target])
					queue.push(p->target);
				p = p->next;
			}
		}
		std::cout << "\n";
	}
	delete[] visit;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::topological_sort() const
{
	LinkedQueue<int> queue;
	int* degree = new int[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
	{
		degree[i] = 0;
		for (EdgeNode* p = vertex[i].head; p != nullptr; p = p->next)
			degree[p->target]++;
	}
	for (int i = 0; i < this->total_vertex; i++)
		if (degree[i] == 0)
			queue.push(i);
	std::cout << "Topological sort: ";
	while (!queue.empty())
	{
		int current = queue.pop();
		std::cout << vertex[current].value << " ";
		for (EdgeNode* p = vertex[current].head; p != nullptr; p = p->next)
			if (--degree[p->target] == 0)
				queue.push(p->target);
	}
	std::cout << "\n";
	delete[] degree;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::critical_path() const
{
	EdgeType* early = new EdgeType[this->total_vertex], * late = new EdgeType[this->total_vertex];
	int* topology = new int[this->total_vertex], * degree = new int[this->total_vertex];
	LinkedQueue<int> queue;
	for (int i = 0; i < this->total_vertex; i++)
	{
		degree[i] = 0;
		for (EdgeNode* p = vertex[i].head; p != nullptr; p = p->next)
			degree[p->target]++;
	}
	for (int i = 0; i < this->total_vertex; i++)
		if (degree[i] == 0)
			queue.push(i);
	for (int i = 0; i < this->total_vertex; i++)
		topology[i] = 0;
	int k = 0;
	while (k < this->total_vertex && !queue.empty())
	{
		topology[k] = queue.pop();
		for (EdgeNode* p = vertex[topology[k]].head; p != nullptr; p = p->next)
			if (--degree[p->target] == 0)
				queue.push(p->target);
		k++;
	}
	for (int i = 0; i < this->total_vertex; i++)
		early[i] = 0;
	for (int i = 0; i < this->total_vertex; i++)
		for (EdgeNode* p = vertex[topology[i]].head; p != nullptr; p = p->next)
			if (early[p->target] < early[topology[i]] + p->weight)
				early[p->target] = early[topology[i]] + p->weight;
	for (int i = 0; i < this->total_vertex; i++)
		late[i] = early[this->total_vertex - 1];
	for (int i = this->total_vertex - 1; i >= 0; i--)
		for (EdgeNode* p = vertex[topology[i]].head; p != nullptr; p = p->next)
			if (late[topology[i]] > late[p->target] - p->weight)
				late[topology[i]] = late[p->target] - p->weight;
	std::cout << "Critical path: ";
	for (int i = 0; i < this->total_vertex; i++)
		if (early[topology[i]] == late[topology[i]])
			std::cout << "(" << vertex[topology[i]].value << ", " << early[topology[i]] << ")\t";
	std::cout << "\n";
	delete[] early;
	delete[] late;
	delete[] topology;
	delete[] degree;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::kruskal() const
{
	PriorityQueue<EdgeInfo> heap;
	FindUnionSet set(this->total_vertex);
	for (int i = 0; i < this->total_vertex; i++)
		for (EdgeNode* p = vertex[i].head; p != nullptr; p = p->next)
			if (i < p->target)
				heap.push(EdgeInfo(i, p->target, p->weight));
	std::cout << "Minimum spanning tree:\t";
	int connect = 0;
	while (connect < this->total_vertex - 1)
	{
		EdgeInfo e = heap.pop();
		int u = set.find(e.begin), v = set.find(e.end);
		if (u != v)
		{
			connect++;
			set.merge(u, v);
			std::cout << "(" << vertex[e.begin].value << ", " << vertex[e.end].value << ")\t";
		}
	}
	std::cout << "\n";
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::prim() const
{
	PriorityQueue<EdgeInfo> heap;
	bool* visit = new bool[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
		visit[i] = false;
	visit[0] = true;
	for (EdgeNode* p = vertex[0].head; p != nullptr; p = p->next)
		heap.push(EdgeInfo(0, p->target, p->weight));
	std::cout << "Minimum spanning tree:\t";
	int connect = 0;
	while (connect < this->total_vertex - 1)
	{
		EdgeInfo e = heap.pop();
		int u = e.begin, v = e.end;
		if (visit[v])
			continue;
		connect++;
		visit[v] = true;
		for (EdgeNode* p = vertex[v].head; p != nullptr; p = p->next)
			if (!visit[p->target])
				heap.push(EdgeInfo(v, p->target, p->weight));
		std::cout << "(" << vertex[u].value << ", " << vertex[v].value << ")\t";
	}
	std::cout << "\n";
	delete[] visit;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::unweighted_distance(VertexType start, EdgeType infinity) const
{
	LinkedQueue<int> queue;
	EdgeType* distance = new EdgeType[this->total_vertex];
	int* precursor = new int[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
		distance[i] = infinity, precursor[i] = -1;
	int s = position(start);
	distance[s] = 0, precursor[s] = s;
	queue.push(s);
	while (!queue.empty())
	{
		int u = queue.pop();
		for (EdgeNode* p = vertex[u].head; p != nullptr; p = p->next)
			if (distance[p->target] == infinity)
			{
				distance[p->target] = distance[u] + 1, precursor[p->target] = u;
				queue.push(p->target);
			}
	}
	for (int i = 0; i < this->total_vertex; i++)
	{
		std::cout << "Unweighted distance form " << start << " to " << vertex[i].value << " is " << distance[i] << ": ";
		print_path(s, i, precursor);
		std::cout << "\n";
	}
	delete[] distance;
	delete[] precursor;
}

template<typename VertexType, typename EdgeType>
inline void AdjacencyListGraph<VertexType, EdgeType>::dijkstra(VertexType start, EdgeType infinity) const
{
	typedef pair<int, EdgeType> NodeInfo;
	PriorityQueue<NodeInfo> heap;
	EdgeType* distance = new EdgeType[this->total_vertex];
	int* precursor = new int[this->total_vertex];
	bool* visit = new bool[this->total_vertex];
	for (int i = 0; i < this->total_vertex; i++)
		distance[i] = infinity, visit[i] = false, precursor[i] = -1;
	int s = position(start);
	distance[s] = 0, precursor[s] = s;
	heap.push(NodeInfo(0, s));
	while (!heap.empty())
	{
		int u = heap.pop().second;
		if (visit[u])
			continue;
		visit[u] = true;
		for (EdgeNode* p = vertex[u].head; p != nullptr; p = p->next)
		{
			int v = p->target;
			if (distance[v] > distance[u] + p->weight)
			{
				distance[v] = distance[u] + p->weight, precursor[v] = u;
				heap.push(NodeInfo(distance[v], v));
			}
		}
	}
	for (int i = 0; i < this->total_vertex; i++)
	{
		std::cout << "Shortest path form " << start << " to " << vertex[i].value << " is:\t";
		print_path(s, i, precursor);
		std::cout << "\twith length of " << distance[i] << "\n";
	}
	delete[] distance;
	delete[] precursor;
	delete[] visit;
}