
numbers = [
    ((i * 37) % 1000)
    for i in range(5000)
]

dp = [1] * len(numbers)

for i in range(len(numbers)):
    for j in range(i):
        if numbers[j] < numbers[i]:
            dp[i] = max(dp[i], dp[j] + 1)

print(max(dp))
