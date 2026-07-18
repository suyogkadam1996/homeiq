import modal

app = modal.App("homeiq-price-predictor")
image = modal.Image.debian_slim().pip_install("scikit-learn", "xgboost", "datasets", "numpy")

@app.function(image=image)
def train_and_predict(description: str) -> str:
    from datasets import load_dataset
    from sklearn.feature_extraction.text import CountVectorizer
    from xgboost import XGBRegressor

    # Reload and clean the same data as Sprint 6
    dataset = load_dataset("mcauley-lab/amazon-reviews-2023", "raw_meta_All_Beauty", split="full")
    prices, descriptions = [], []
    seen = set()
    for item in dataset:
        price_str = item["price"]
        if price_str is not None and price_str != "None":
            try:
                price = float(price_str)
                if price > 0 and item["title"] not in seen:
                    seen.add(item["title"])
                    prices.append(price)
                    descriptions.append(item["title"])
            except (ValueError, TypeError):
                continue

    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(descriptions)
    model = XGBRegressor(random_state=42)
    model.fit(X, prices)

    new_input = vectorizer.transform([description])
    prediction = model.predict(new_input)[0]
    return f"Estimated price: ${prediction:.2f}"

@app.local_entrypoint()
def main():
    result = train_and_predict.remote("Wireless mechanical keyboard, RGB backlit")
    print(result)