
text = "distributed systems and artificial intelligence " * 500_000

count = 0

for word in text.split():
    if word.startswith("a"):
        count += 1

print(count)
