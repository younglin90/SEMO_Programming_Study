// This example demonstrates filtering and transforming a range on the
// fly with view adaptors.

#include <iostream>
#include <string>
#include <vector>
#include <cassert>


#include <range/v3/all.hpp>
using std::cout;



struct Boundary {
    size_t startFace, nFaces;
};
struct Face {
    size_t id;
};

template<typename T>
class SlicedRange : public ranges::view_facade<SlicedRange<T>> {
    friend ranges::range_access;

    T& read() const { return (*datas_)[current_index_]; }
    bool equal(ranges::default_sentinel_t) const {
        return current_index_ - start_ == n_;
    }
    void next() { ++current_index_; }

    std::vector<T>* datas_;
    std::size_t start_, n_;
    std::size_t current_index_;

public:
    SlicedRange() = delete;
    SlicedRange(std::vector<T>& datas, const size_t& start, const size_t& n) :
        datas_(&datas),
        n_(n),
        start_(start),
        current_index_(start)
    { }
};


template<typename T>
class IndexedRange : public ranges::view_facade<IndexedRange<T>> {
    friend ranges::range_access;

    T& read() const { return (*datas_)[indexed_[current_index_]]; }
    bool equal(ranges::default_sentinel_t) const {
        return current_index_ == indexed_.size();
    }
    void next() { ++current_index_; }

    std::vector<T>* datas_;
    std::vector<size_t> indexed_;
    std::size_t current_index_ = 0;

public:
    IndexedRange() = delete;
    IndexedRange(std::vector<T>& datas, const std::vector<size_t>& indexed) :
        datas_(&datas),
        indexed_(indexed),
        current_index_(0)
    { }
};

int main()
{
    std::vector<Face> faces = { {1}, {2}, {3}, {4}, {5} };
    Boundary bc;
    bc.startFace = 1;
    bc.nFaces = 3;

    SlicedRange sliced_boundaries(faces, bc.startFace, bc.nFaces);

    std::vector<size_t> indexed = { 3, 0, 2, 1 };
    IndexedRange indexed_boundaries(faces, indexed);

    // current_index() 값을 루프 내에서 확인하기 위해 범위 기반 반복 대신 명시적 반복자를 사용
    for (const auto& b : indexed_boundaries) {
        std::cout << "boundary id: " << b.id << std::endl;
    }

    auto squares = indexed_boundaries | 
        ranges::views::transform([](const Face& i) { return i.id; }) | 
        ranges::views::transform([](const size_t& i) { return i; });
    std::vector<size_t> squares_results = squares | ranges::to<std::vector<size_t>>();

    //ranges::for_each(indexed_boundaries, [](const Face& i) { return i.id * i.id; });
    //for (const auto& b : indexed_boundaries) {
    //    std::cout << "boundary id: " << b.id << std::endl;
    //}

    for (const auto& b : squares_results) {
        std::cout << b << std::endl;
    }

    faces = { {2}, {3}, {4}, {5}, {6} };

    for (const auto& b : squares_results) {
        std::cout << b << std::endl;
    }
}