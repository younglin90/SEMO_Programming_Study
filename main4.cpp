#include <vector>
#include <array>
#include <iostream>
struct Test {
    size_t size = 0;

    constexpr void insert() {
        ++size;
    }
};

int main() {

    constexpr Test test = []()constexpr {
        Test t{ 0 };
        ++t.size;
        t.insert();
        t.insert();
        t.insert();
        return t;
        }();

    constexpr auto aa = test.size;
    static_assert(aa == 4);


}