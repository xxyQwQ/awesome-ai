#pragma once

template <typename VertexType, typename EdgeType>
class Graph
{
protected:
	int total_vertex = 0, total_edge = 0;
public:
	virtual void insert(VertexType x, VertexType y, EdgeType z) = 0;
	virtual void remove(VertexType x, VertexType y) = 0;
	virtual bool exist(VertexType x, VertexType y) const = 0;
	int vertex_count() const
	{
		return total_vertex;
	}
	int edge_count() const
	{
		return total_edge;
	}
};