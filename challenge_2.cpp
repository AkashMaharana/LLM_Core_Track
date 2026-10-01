#include <iostream>
#include <string>

int main() {
    const std::string pattern = "distributed systems and artificial intelligence ";
    const int repetitions = 500000;

    // Count words starting with 'a' in one pattern
    int countPerPattern = 0;
    std::size_t pos = 0;
    const std::size_t len = pattern.size();

    while (pos < len) {
        // Skip spaces
        while (pos < len && pattern[pos] == ' ') {
            ++pos;
        }
        if (pos >= len) break;

        // Start of word
        bool startsWithA = (pattern[pos] == 'a');

        // Move to end of word
        while (pos < len && pattern[pos] != ' ') {
            ++pos;
        }

        if (startsWithA) {
            ++countPerPattern;
        }
    }

    long long totalCount = static_cast<long long>(countPerPattern) *
                           static_cast<long long>(repetitions);

    std::cout << totalCount << std::endl;
    return 0;
}