#include <iostream>
#include <cmath>
#include <cstdint>

int main() {
    std::int32_t count = 0;

    for (std::int32_t number = 2; number < 200000; ++number) {
        bool is_prime = true;

        int limit = static_cast<int>(std::sqrt(static_cast<double>(number))) + 1;

        for (int divisor = 2; divisor < limit; ++divisor) {
            if (number % divisor == 0) {
                is_prime = false;
                break;
            }
        }

        if (is_prime) {
            ++count;
        }
    }

    std::cout << count << '\n';
    return 0;
}