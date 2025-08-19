#include <tuple>
#include <iterator>
#include <utility>
#include <vector>
#include <iostream>
#include <ranges>
#include <list>

namespace semo::ranges {

    // @brief range 구조체
    template <class It>
    struct range {
        It m_begin;
        It m_end;

        constexpr range(It begin, It end)
            : m_begin(std::move(begin)), m_end(std::move(end))
        {
            //std::cout << SEMO_GetTypeName(It) << std::endl;
        }

        constexpr It begin() { return m_begin; }
        constexpr It end() { return m_end; }


    };
    // the deduction rule 정의
    template <class It>
    range(It begin, It end) -> range<It>;


    // @brief range를 받아서 range를 리턴하는 함수 객체
    template <class F>
    struct pipable
    {
        F m_f;

        constexpr pipable(F f)
            : m_f(std::move(f))
        {
        }

        // function style: map(func)(range)
        template <class ...Rs>
        constexpr decltype(auto) operator()(Rs &&...rs) {
            return m_f(range(rs.begin(), rs.end())...);
        }

        // pipe style: range | map(func)
        template <class R>
        constexpr friend decltype(auto) operator|(R&& r, pipable& self) {
            return self(std::forward<decltype(r)>(r));
        }
    };

    // @brief pipable을 받아서 pipable을 리턴하는 함수 객체
    template <class Func, class Base>
    struct map_iterator {
        Func m_func;
        Base m_it;

        // using `decltype(auto)` (since C++14) instead of `auto` so that
        // even if `m_func` returns a reference, it automatically becames
        // `auto &` rather than dereferencing that...
        constexpr decltype(auto) operator*() const {
            return m_func(*m_it);
        }

        constexpr map_iterator& operator++() {
            m_it++;
            return *this;
        }

        constexpr bool operator!=(map_iterator const& that) const {
            return m_it != that.m_it;
        }
    };
    template <class Func, class Base>
    map_iterator(Func, Base) -> map_iterator<Func, Base>;

    // @brief map 함수
    template <class F>
    static constexpr auto map(F&& f) {
        return pipable([=](auto&& r) {
            return range
            (map_iterator{ f, r.begin() }
                , map_iterator{ f, r.end() }
            );
            });
    }

    // @brief enumerate 반복자
    template <class Base>
    struct enumerate_iterator {
        Base m_it;
        std::size_t m_index = 0;
        using value_type = std::pair<std::size_t, std::remove_cvref_t<decltype(*std::declval<Base>())>>;
        value_type m_value;

        constexpr enumerate_iterator(Base it)
            : m_it(std::move(it))
        {
        }

        constexpr auto operator*() {
            m_value.first = m_index;
            m_value.second = *m_it;
            return std::tie(m_index, *m_it);
        }

        constexpr enumerate_iterator& operator++() {
            ++m_it;
            ++m_index;
            return *this;
        }
        constexpr enumerate_iterator& operator++(int) {
            auto tmp = *this;
            ++m_it;
            ++m_index;
            return tmp;
        }

        constexpr bool operator!=(enumerate_iterator& that) {
            return m_it != that.m_it;
        }
    };
    template <class Base>
    enumerate_iterator(Base) -> enumerate_iterator<Base>;

    // @brief enumerate 함수
    static auto enumerate = pipable([](auto&& r) {
        return range
        (enumerate_iterator(r.begin())
            , enumerate_iterator(r.end())
        );
        });



    // @brief zip 반복자
    template <class... Iterators>
    struct zip_iterator {

        using iter_type = std::tuple<Iterators...>;
        //using value_type = std::tuple<std::remove_cvref_t<decltype(*std::declval<Bases>())>... >;
        using value_type = std::tuple<
            std::remove_cvref_t<decltype(*std::declval<Iterators>())>
            ... >;
        iter_type m_it;
        value_type m_value;

        using vluae_ref_type = std::tuple<
            decltype(*std::declval<Iterators>())&
            ... >;

        zip_iterator(Iterators... itors) : m_it(itors...) {}

        constexpr decltype(auto) operator*() {

            //return std::tie(*std::get<0>(m_it), *std::get<1>(m_it));
            //m_value = std::apply([](auto&... args) { return std::make_tuple((*args)...); }, m_it);

            //std::tuple<decltype(*std::declval<Bases>())&...> m_ref_value
            //    std::apply([](auto&... args) { return std::make_tuple((*args)...); }, m_it);

            auto refTuple = std::apply([](auto&... args) {
                return std::make_tuple(std::ref(*args)...);
                }, m_it);


            return refTuple;
        }

        constexpr zip_iterator& operator++() {
            std::apply([](auto&... args) { (++args, ...); }, m_it);
            return *this;
        }

        constexpr bool operator!=(zip_iterator const& that) const {
            return std::get<0>(m_it) != std::get<0>(that.m_it);
        }

    };
    template <class... Iterators>
    zip_iterator(Iterators...) -> zip_iterator<Iterators...>;


    // @brief zip 함수 정의
    template<class Count, class... Conts>
    static constexpr auto zip(Count&& f0, Conts&&... f) {
        if (!(... && (f.size() == f0.size()))) {
            std::cout << "Container sizes must be the same" << std::endl;
            //assert(false);
            //throw std::invalid_argument("Container sizes must be the same");
        }
        auto aa = range(zip_iterator(f0.begin(), f.begin()...), zip_iterator(f0.end(), f.end()...));
        return aa;
    };



}






//
//template<typename Container>
//class Enumerator {
//private:
//    Container& container;
//    using iterator_type = decltype(std::begin(std::declval<Container&>()));
//
//    class Iterator {
//    private:
//        iterator_type it;
//        std::size_t index;
//
//    public:
//        Iterator(iterator_type iterator, std::size_t idx)
//            : it(iterator), index(idx) {
//        }
//
//        bool operator!=(const Iterator& other) const {
//            return it != other.it;
//        }
//
//        void operator++() {
//            ++it;
//            ++index;
//        }
//
//        auto operator*() {
//            return std::make_pair(index, std::ref(*it));
//        }
//    };
//
//public:
//    explicit Enumerator(Container& cont) : container(cont) {}
//
//    auto begin() {
//        return Iterator(std::begin(container), 0);
//    }
//
//    auto end() {
//        return Iterator(std::end(container), 0);
//    }
//};
//
//// 편의를 위한 헬퍼 함수
//template<typename Container>
//auto enumerate(Container& container) {
//    return Enumerator<Container>(container);
//}





template <typename... Containers>
class Zipper {
private:
    std::tuple<Containers&...> containers;

    template <typename... Iters>
    class Iterator {
    private:
        std::tuple<Iters...> iters;

    public:
        Iterator(Iters... its) : iters(its...) {}

        auto operator*() {
            return std::apply([](auto&... its) {
                return std::tuple<decltype(*its)...>(*its...);
                }, iters);
        }

        Iterator& operator++() {
            std::apply([](auto&... its) { (++its, ...); }, iters);
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return iters != other.iters;
        }
    };

public:
    explicit Zipper(Containers&... conts) : containers(conts...) {}

    auto begin() {
        return std::apply([](auto&... conts) {
            return Iterator(std::begin(conts)...);
            }, containers);
    }

    auto end() {
        return std::apply([](auto&... conts) {
            return Iterator(std::end(conts)...);
            }, containers);
    }
};

template <typename... Containers>
auto zip(Containers&... containers) {
    return Zipper<Containers...>(containers...);
}

template <typename Container>
class Enumerator {
private:
    Container container;

    template <typename Iter>
    class Iterator {
    private:
        size_t index;
        Iter iter;

    public:
        Iterator(size_t idx, Iter it) : index(idx), iter(it) {}

        auto operator*() {
            return std::tuple<size_t, decltype(*iter)>(index, *iter);
        }

        Iterator& operator++() {
            ++index;
            ++iter;
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return iter != other.iter;
        }
    };

public:
    explicit Enumerator(Container cont) : container(cont) {}

    auto begin() {
        return Iterator(0, container.begin());
    }

    auto end() {
        return Iterator(0, container.end());
    }
};

template <typename Container>
auto enumerate(Container&& container) {
    return Enumerator<Container>(std::forward<Container>(container));
}

//
//
//template<typename Container>
//class Enumerator {
//private:
//    Container& container;
//    using iterator_type = decltype(std::begin(std::declval<Container&>()));
//
//    class Iterator {
//    private:
//        iterator_type it;
//        std::size_t index;
//
//    public:
//        Iterator(iterator_type iterator, std::size_t idx)
//            : it(iterator), index(idx) {
//        }
//
//        bool operator!=(const Iterator& other) const {
//            return it != other.it;
//        }
//
//        void operator++() {
//            ++it;
//            ++index;
//        }
//
//        auto operator*() {
//            return std::make_pair(index, std::ref(*it));
//        }
//    };
//
//public:
//    explicit Enumerator(Container& cont) : container(cont) {}
//
//    auto begin() {
//        return Iterator(std::begin(container), 0);
//    }
//
//    auto end() {
//        return Iterator(std::end(container), 0);
//    }
//};
//
//// 편의를 위한 헬퍼 함수
//template<typename Container>
//auto enumerate(Container& container) {
//    return Enumerator<Container>(container);
//}





int main() {
    std::vector<int> A = { 1, 2, 3 };
    std::vector<int> B = { 4, 5, 6 };
    std::vector<int> C = { 7, 8, 9 };

    // enumerate와 zip을 함께 사용
    for (auto [a, b, c] : zip(A, B, C)) {
        b += 100;
    }
    for (auto& b : B) {
        std::cout << b << std::endl;
    }
    for (auto [i, tpl] : enumerate(zip(A, B, C))) {
        auto& [a, b, c] = tpl;  // zip에서 반환된 튜플을 풀어냄
        std::cout << "인덱스: " << i << ", A: " << a << ", B: " << b << ", C: " << c << '\n';
        b += 100;
    }
    for (auto& b : B) {
        std::cout << b << std::endl;
    }

    //for (auto [word0, word1, word2] : enumerate(semo::ranges::zip(words, words2))) {
    //    std::cout << word0 << ": " << word1 << '\n';
    //    word1 += "++++++";
    //}
    //for (auto [word0, word1] : semo::ranges::zip(words, words2)) {
    //    std::cout << word0 << ": " << word1 << '\n';
    //}

    // 값 수정이 가능한 열거
    //for (auto [idx, elem] : enumerate(words)) {
    //    std::cout << idx << ": " << elem << '\n';
    //    elem += "++++++";
    //}
    //for (auto [idx, elem] : semo::ranges::enumerate(words)) {
    //    std::cout << idx << ": " << elem << '\n';
    //    elem += "++++++";
    //}

    //// 수정된 결과 출력
    //for (const auto& word : words) {
    //    std::cout << word << '\n';
    //}

    return 0;
}