from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset("mcauley-lab/amazon-reviews-2023", "raw_meta_All_Beauty", split="full")
print(f"Total items: {len(dataset)}")

prices = []
descriptions = []
for item in dataset:
    price_str = item["price"]
    if price_str is not None and price_str != "None":   # catch BOTH the real empty value AND the text "None"
        try:
            price = float(price_str)
            if price > 0:   # also reject accidental $0 entries
                prices.append(price)
                descriptions.append(item["title"])
        except (ValueError, TypeError):
            continue   # skip anything that isn't a real number at all

print(f"Items with a real, usable price: {len(prices)}")
print(f"Example price: {prices[0]}, example description: {descriptions[0]}")

plt.hist(prices, bins=50)
plt.title("Price distribution")
plt.xlabel("Price ($)")
plt.ylabel("Number of items")
plt.savefig("price_distribution.png")
print("Saved chart to price_distribution.png")

# Remove duplicate descriptions - otherwise a model can "cheat" by memorizing repeats
seen = set()
unique_prices = []
unique_descriptions = []
for price, desc in zip(prices, descriptions):
    if desc not in seen:
        seen.add(desc)
        unique_prices.append(price)
        unique_descriptions.append(desc)

print(f"\nBefore removing duplicates: {len(prices)}")
print(f"After removing duplicates: {len(unique_prices)}")