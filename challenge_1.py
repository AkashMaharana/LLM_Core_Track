
import math

count = 0

for number in range(2, 200_000):
    is_prime = True

    limit = int(math.sqrt(number)) + 1

    for divisor in range(2, limit):
        if number % divisor == 0:
            is_prime = False
            break

    if is_prime:
        count += 1

print(count)
