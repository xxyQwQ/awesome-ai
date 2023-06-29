#pragma once

template <typename ElementType>
class Less
{
public:
	bool operator()(const ElementType& left, const ElementType& right) const
	{
		return left < right;
	}
};

template <typename ElementType>
class Greater
{
public:
	bool operator()(const ElementType& left, const ElementType& right) const
	{
		return left > right;
	}
};