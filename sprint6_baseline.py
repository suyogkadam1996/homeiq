from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

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

print(f"Working with {len(prices)} clean, unique, priced items")

X_train, X_test, y_train, y_test = train_test_split(descriptions, prices, test_size=0.2, random_state=42)
print(f"Training on {len(X_train)} items, testing on {len(X_test)} items")

average_price = np.mean(y_train)
baseline_predictions = [average_price] * len(y_test)
baseline_error = np.mean(np.abs(np.array(baseline_predictions) - np.array(y_test)))
print(f"\nBaseline (always guess ${average_price:.2f}): average error = ${baseline_error:.2f}")

vectorizer = CountVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
}

print()
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)
    error = np.mean(np.abs(predictions - y_test))
    print(f"{name}: average error = ${error:.2f}")