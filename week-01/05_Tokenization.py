import tiktoken

def main():
	encoding = tiktoken.get_encoding("o200k_base")

	text1 = "The battery life is excellent."
	tex2 = "https://www.tesco.com/groceries/en-GB/products/268566236"
	text3 = "TPNB-62678007-UK-2026"
	text4 ="""
	{
		"tpnb": "62678007",
		"sentiment": "Taste",
		"score": 0.91
	}
	""" 

	text = text3
	
	token_ids = encoding.encode(text)
	
	print("Text:")
	print(text)
	print("\nToken IDs:")
	print(token_ids)
	print("\nTokens:")
	for token_id in token_ids:
		print(repr(encoding.decode([token_id])))

	print("\nTotal tokens:", len(token_ids))


if __name__ == "__main__":
	main()