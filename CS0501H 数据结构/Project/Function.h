#pragma once
#include <cstdlib>

const int INF = 0x7fffffff;

template<typename FirstType, typename SecondType>
class pair
{
public:
	FirstType first;
	SecondType second;
	pair() : first(), second() {}
	pair(FirstType _first, SecondType _second) : first(_first), second(_second) {}
	bool operator==(const pair<FirstType, SecondType>& another) const
	{
		return first == another.first && second == another.second;
	}
	bool operator<(const pair<FirstType, SecondType>& another) const
	{
		return first == another.first ? second < another.second : first < another.first;
	}
	bool operator>(const pair<FirstType, SecondType>& another)
	{
		return first == another.first ? second > another.second : first > another.first;
	}
};

template <typename ElementType>
ElementType max(const ElementType& x, const ElementType& y)
{
	return (x > y) ? x : y;
}

template <typename ElementType>
ElementType min(const ElementType& x, const ElementType& y)
{
	return (x < y) ? x : y;
}

template <typename ElementType>
void swap(ElementType& x, ElementType& y)
{
	ElementType t = x;
	x = y, y = t;
}

template <typename ElementType>
void random_shuffle(ElementType data[], int size)
{
	for (int i = size - 1; i; i--)
	{
		int p = rand() % (i + 1);
		swap(data[i], data[p]);
	}
}

template <typename ElementType>
void insert_sort(ElementType data[], int size)
{
	ElementType temp;
	for (int i = 1, j; i < size; i++)
	{
		temp = data[i];
		for (j = i; j && temp < data[j - 1]; j--)
			data[j] = data[j - 1];
		data[j] = temp;
	}
}

template <typename ElementType>
void shell_sort(ElementType data[], int size)
{
	ElementType temp;
	for (int step = size / 2; step; step >>= 1)
		for (int i = step, j; i < size; i++)
		{
			temp = data[i];
			for (j = i - step; j >= 0 && temp < data[j]; j -= step)
				data[j + step] = data[j];
			data[j + step] = temp;
		}
}

template <typename ElementType>
void select_sort(ElementType data[], int size)
{
	for (int i = 0, p; i < size - 1; i++)
	{
		p = i;
		for (int j = i + 1; j < size; j++)
			if (data[j] < data[p])
				p = j;
		swap(data[i], data[p]);
	}
}

template <typename ElementType>
void bubble_sort(ElementType data[], int size)
{
	bool flag = true;
	for (int i = 1; i < size && flag; i++)
	{
		flag = false;
		for (int j = 0; j < size - i; j++)
			if (data[j + 1] < data[j])
			{
				swap(data[j], data[j + 1]);
				flag = true;
			}
	}
}

template <typename ElementType>
void heap_update(ElementType data[], int hole, int size)
{
	ElementType temp = data[hole];
	for (int next = 0; hole * 2 + 1 < size; hole = next)
	{
		next = hole * 2 + 1;
		if (next != size - 1 && data[next + 1] > data[next])
			next++;
		if (data[next] > temp)
			data[hole] = data[next];
		else
			break;
	}
	data[hole] = temp;
}

template <typename ElementType>
void heap_sort(ElementType data[], int size)
{
	for (int i = size / 2 - 1; i >= 0; i--)
		heap_update(data, i, size);
	for (int i = size - 1; i; i--)
	{
		swap(data[0], data[i]);
		heap_update(data, 0, i);
	}
}

template <typename ElementType>
int make_partition(ElementType data[], int low, int high)
{
	ElementType key = data[low];
	do
	{
		while (low < high && data[high] >= key)
			high--;
		if (low < high)
			data[low++] = data[high];
		while (low < high && data[low] <= key)
			low++;
		if (low < high)
			data[high--] = data[low];
	} while (low < high);
	data[low] = key;
	return low;
}

template <typename ElementType>
void quick_sort(ElementType data[], int low, int high)
{
	if (low >= high)
		return;
	int split = make_partition(data, low, high);
	quick_sort(data, low, split - 1);
	quick_sort(data, split + 1, high);
}

template <typename ElementType>
void quick_sort(ElementType data[], int size)
{
	quick_sort(data, 0, size - 1);
}

template <typename ElementType>
void make_merger(ElementType data[], int left, int split, int right)
{
	ElementType* temp = new ElementType[right - left + 1];
	int i = left, j = split, k = 0;
	while (i < split && j <= right)
	{
		if (data[i] < data[j])
			temp[k++] = data[i++];
		else
			temp[k++] = data[j++];
	}
	while (i < split)
		temp[k++] = data[i++];
	while (j <= right)
		temp[k++] = data[j++];
	for (i = 0, k = left; k <= right;)
		data[k++] = temp[i++];
	delete[] temp;
}

template <typename ElementType>
void merge_sort(ElementType data[], int left, int right)
{
	if (left == right)
		return;
	int split = (left + right) / 2;
	merge_sort(data, left, split);
	merge_sort(data, split + 1, right);
	make_merger(data, left, split + 1, right);
}

template <typename ElementType>
void merge_sort(ElementType data[], int size)
{
	merge_sort(data, 0, size - 1);
}