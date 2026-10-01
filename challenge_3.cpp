#include <bits/stdc++.h>
using namespace std;

int main() {
    const int N = 5000;
    const int MOD = 1000;
    const int MULT = 37;

    vector<int> numbers;
    numbers.reserve(N);
    for (int i = 0; i < N; ++i) {
        numbers.push_back((static_cast<long long>(i) * MULT) % MOD);
    }

    vector<int> dp(N, 1);

    for (int i = 0; i < N; ++i) {
        int best = 1;
        int ni = numbers[i];
        for (int j = 0; j < i; ++j) {
            if (numbers[j] < ni) {
                int candidate = dp[j] + 1;
                if (candidate > best) best = candidate;
            }
        }
        dp[i] = best;
    }

    int result = 0;
    for (int v : dp) {
        if (v > result) result = v;
    }

    cout << result << '\n';
    return 0;
}