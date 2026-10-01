#include <iostream>
#include <cstdint>

int main() {
    std::int64_t total = 0;

    for (std::int64_t i = 0; i < 10000000; ++i) {
        std::int64_t mod_i = i % 97;
        total += (mod_i * mod_i) % 97;
    }

    std::cout << total << '\n';
    return 0;
}