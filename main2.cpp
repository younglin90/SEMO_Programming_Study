#include <tuple>
#include <iterator>
#include <utility>
#include <vector>
#include <iostream>
#include <ranges>
#include <list>
//
//// enumerate 함수 템플릿
//template <typename T>
//constexpr auto enumerate(T&& iterable) {
//    using std::begin;
//    using std::end;
//
//    struct iterator {
//        size_t i;
//        decltype(begin(iterable)) iter;
//
//        bool operator!=(const iterator& other) const { return iter != other.iter; }
//        void operator++() { ++i; ++iter; }
//        auto operator*() const { return std::tie(i, *iter); }
//    };
//
//    struct iterable_wrapper {
//        T iterable;
//
//        auto begin() { return iterator{ 0, std::begin(iterable) }; }
//        auto end() { return iterator{ 0, std::end(iterable) }; }
//    };
//
//    return iterable_wrapper{ std::forward<T>(iterable) };
//}
//
//
//
//int main() {
//    std::vector<int> vec = { 10, 20, 30, 40 };
//
//    // enumerate를 사용하여 인덱스와 값을 동시에 출력
//    for (auto [i, value] : enumerate(vec)) {
//        std::cout << "Index: " << i << ", Value: " << value << '\n';
//    }
//
//    return 0;
//}


//
//// Efficient enumerate implementation for C++20
//template <typename Container>
//class Enumerate {
//private:
//    Container& container_;
//
//    // Iterator wrapper that provides index and reference to element
//    template <typename Iterator>
//    class EnumerateIterator {
//    private:
//        size_t index_;
//        Iterator iterator_;
//
//    public:
//        using iterator_category = std::forward_iterator_tag;
//        using value_type = std::tuple<size_t, decltype(*std::declval<Iterator>())>;
//        using difference_type = std::ptrdiff_t;
//        using pointer = value_type*;
//        using reference = value_type&;
//
//        EnumerateIterator(size_t index, Iterator iterator)
//            : index_(index), iterator_(iterator) {
//        }
//
//        // Prefix increment
//        EnumerateIterator& operator++() {
//            ++index_;
//            ++iterator_;
//            return *this;
//        }
//
//        // Postfix increment
//        EnumerateIterator operator++(int) {
//            EnumerateIterator temp = *this;
//            ++index_;
//            ++iterator_;
//            return temp;
//        }
//
//        // Dereference operator returns tuple of (index, reference to element)
//        value_type operator*() const {
//            return std::tie(index_, *iterator_);
//        }
//
//        // Comparison operators
//        bool operator!=(const EnumerateIterator& other) const {
//            return iterator_ != other.iterator_;
//        }
//
//        bool operator==(const EnumerateIterator& other) const {
//            return iterator_ == other.iterator_;
//        }
//    };
//
//public:
//    // Constructor taking container by reference
//    explicit Enumerate(Container& container) : container_(container) {}
//
//    // Begin iterator
//    auto begin() {
//        return EnumerateIterator<typename Container::iterator>(0, container_.begin());
//    }
//
//    // End iterator
//    auto end() {
//        return EnumerateIterator<typename Container::iterator>(container_.size(), container_.end());
//    }
//};
//
//// Convenience function to create Enumerate
//template <typename Container>
//auto enumerate(Container& container) {
//    return Enumerate<Container>(container);
//}
//
//int main() {
//    // Test with vector of integers
//    std::vector<int> vec = { 10, 20, 30, 40 };
//
//    std::cout << "Vector test:\n";
//    for (auto [index, value] : enumerate(vec)) {
//        std::cout << "Index: " << index << ", Value: " << value << '\n';
//
//        // Modify value to show reference semantics work
//        value *= 2;
//    }
//
//    std::cout << "Modified vector:\n";
//    for (int v : vec) {
//        std::cout << v << ' ';
//    }
//    std::cout << '\n';
//
//    // Test with list of strings
//    std::list<std::string> names = { "Alice", "Bob", "Charlie" };
//
//    std::cout << "List test:\n";
//    for (auto [index, name] : enumerate(names)) {
//        std::cout << "Index: " << index << ", Name: " << name << '\n';
//    }
//
//    return 0;
//}



// Efficient enumerate implementation for C++20 with reference support
template <typename Container>
class Enumerate {
private:
    Container& container_;

    // Iterator wrapper that provides index and reference to element
    template <typename Iterator>
    class EnumerateIterator {
    private:
        size_t index_;
        Iterator iterator_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::tuple<size_t, decltype(*iterator_)>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        EnumerateIterator(size_t index, Iterator iterator)
            : index_(index), iterator_(iterator) {
        }

        // Prefix increment
        EnumerateIterator& operator++() {
            ++index_;
            ++iterator_;
            return *this;
        }

        // Postfix increment
        EnumerateIterator operator++(int) {
            EnumerateIterator temp = *this;
            ++index_;
            ++iterator_;
            return temp;
        }

        // Dereference operator returns tuple of (index, reference to element)
        auto operator*() const -> std::tuple<size_t, decltype(*iterator_)&> {
            return std::tie(index_, *iterator_);
        }

        // Comparison operators
        bool operator!=(const EnumerateIterator& other) const {
            return iterator_ != other.iterator_;
        }

        bool operator==(const EnumerateIterator& other) const {
            return iterator_ == other.iterator_;
        }
    };

public:
    // Constructor taking container by reference
    explicit Enumerate(Container& container) : container_(container) {}

    // Begin iterator
    auto begin() {
        return EnumerateIterator<typename Container::iterator>(0, container_.begin());
    }

    // End iterator
    auto end() {
        return EnumerateIterator<typename Container::iterator>(container_.size(), container_.end());
    }
};

// Convenience function to create Enumerate
template <typename Container>
auto enumerate(Container& container) {
    return Enumerate<Container>(container);
}

int main() {
    // Test with vector of integers
    std::vector<int> vec = { 10, 20, 30, 40 };

    std::cout << "Vector test:\n";

    // Use auto& to avoid copying and modify the original elements
    for (auto& [index, value] : enumerate(vec)) {
        std::cout << "Index: " << index << ", Value: " << value << '\n';

        // Modify value to show reference semantics work
        value *= 2;
    }

    std::cout << "Modified vector:\n";
    for (int v : vec) {
        std::cout << v << ' ';
    }
    std::cout << '\n';

    return 0;
}