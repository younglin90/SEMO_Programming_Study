

#include <iostream>
#include <format>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <numeric>
//
//template<typename Args>
//void rearange(
//	const std::function<void()>& sort_func,
//	Args inps...
//) {
//
//	
//	std::sort();
//
//}

template<typename T>
void rearange(
	std::vector<T>& inp, 
	const std::vector<size_t>& thisElem_orgIndex
) {
	std::vector<T> tmp(inp.size());
	std::transform(thisElem_orgIndex.begin(), thisElem_orgIndex.end(), tmp.begin(),
		[&inp](size_t index) {
			return inp[index];
		});
	std::swap(inp, tmp);
}

template<typename T>
void rearange(
	std::vector<std::vector<T>>& inp, 
	const std::vector<size_t>& thisElem_orgIndex,
	const std::vector<size_t>& thisElem_moveToIndex
) {

	std::vector<std::vector<T>> tmp(inp.size());
	for (size_t i = 0; i < thisElem_orgIndex.size(); ++i) {
		size_t index = thisElem_orgIndex[i];
		std::swap(tmp[i], inp[index]);
		for (auto& item : tmp[i]) {
			item = thisElem_moveToIndex[item];
		}
	}
	std::swap(inp, tmp);
}


template<typename T>
std::tuple<std::vector<size_t>, std::vector<size_t>> 
reordering_indices(size_t size, T func)
{

	std::vector<size_t> thisElem_orgIndex(size);
	std::iota(thisElem_orgIndex.begin(), thisElem_orgIndex.end(), 0);
	std::sort(thisElem_orgIndex.begin(), thisElem_orgIndex.end(), func);
	std::vector<size_t> thisElem_moveToIndex(size);
	for (size_t i = 0; i < thisElem_orgIndex.size(); ++i) {
		thisElem_moveToIndex[thisElem_orgIndex[i]] = i;
	}

	return std::tie(thisElem_orgIndex, thisElem_moveToIndex);
}

struct Test {

	std::vector<std::array<double, 3>> pos;
	std::vector<size_t> c2g;
	std::vector<std::vector<size_t>> c2v, c2f;

};

int main() {
	
	Test test;
	test.pos = { {1, 1, 2} , {0, 1, 3}, {0, 0, 0}, {0, 1, 2}, {0, 2, 2} };
	test.c2f = { {0,1,2} , {1,1,2}, {2,1,2}, {3,1,2}, {4,1,2} }; // => 2,3,1,4,0 => 0,1,2,3,4

	auto [thisElem_orgIndex, thisElem_moveToIndex] = 
		reordering_indices(
			test.pos.size(),
			[&test](size_t i1, size_t i2) {
				return test.pos[i1] < test.pos[i2];
			});


	for (auto& item : thisElem_orgIndex) {
		std::cout <<
			std::format("{} ", item);
	}
	std::cout << std::endl;
	for (auto& item : thisElem_moveToIndex) {
		std::cout <<
			std::format("{} ", item);
	}
	std::cout << std::endl;

	
	rearange(test.pos, thisElem_orgIndex);
	rearange(test.c2f, thisElem_orgIndex, thisElem_moveToIndex);


	for (auto& item : test.pos) {
		std::cout <<
			std::format("({},{},{}) ", item[0], item[1], item[2]);
	}
	std::cout << std::endl;

	for (auto& item : test.c2f) {
		std::cout <<
			std::format("({} {} {}) ", item[0], item[1], item[2]);
	}
	std::cout << std::endl;

	//rearange([]() {}, test.c2g, test.c2v, test.c2f);


	return 0;
}