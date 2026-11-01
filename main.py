import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("data/customers.csv")

print("CUSTOMER SEGMENTATION PROJECT")
print("-" * 40)

print("\nFirst 5 Customers:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Select features for clustering
features = df[["AnnualIncome", "SpendingScore"]].copy()

# Handle missing values if present
features = features.fillna(features.median())

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# Display customer groups
print("\nCustomer Segmentation Results:")
print(df[["CustomerID", "AnnualIncome", "SpendingScore", "Cluster"]])

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# Plot customer clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["AnnualIncome"],
    df["SpendingScore"],
    c=df["Cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation Using K-Means")
plt.tight_layout()

# Save graph and results
plt.savefig("output/customer_clusters.png", dpi=300)
df.to_csv("output/segmented_customers.csv", index=False)

print("\nGraph saved to output/customer_clusters.png")
print("Results saved to output/segmented_customers.csv")

#plt.show()


print("\n" + "=" * 40)
print("APRIORI - PURCHASE PATTERN ANALYSIS")
print("=" * 40)

# Load transaction dataset
transactions_df = pd.read_csv("data/transactions.csv")

# Convert product lists into transactions
transactions = [
    products.split(";")
    for products in transactions_df["Products"]
]

# Convert transactions into a one-hot encoded table
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

# Find frequent itemsets
frequent_items = apriori(
    basket,
    min_support=0.2,
    use_colnames=True
)

print("\nFrequent Product Combinations:")
print(frequent_items)

# Generate association rules
if not frequent_items.empty:
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.5
    )

    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: ", ".join(sorted(x))
        )
        rules["consequents"] = rules["consequents"].apply(
            lambda x: ", ".join(sorted(x))
        )

        print("\nAssociation Rules:")
        print(rules[[
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]].round(2))

        rules.to_csv(
            "output/association_rules.csv",
            index=False
        )
    else:
        print("\nNo association rules found.")
else:
    print("\nNo frequent product combinations found.")

# Save frequent itemsets
frequent_items.to_csv(
    "output/frequent_itemsets.csv",
    index=False
)

print("\nApriori analysis completed!")