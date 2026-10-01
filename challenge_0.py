
total = 0

for i in range(10_000_000):
    total += (i * i) % 97

print(total)
