import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load customers: how often they order and how much they spend per order
data = pd.read_csv("customers.csv")
X = data[["orders_per_month", "avg_order_value"]]

# 2. Put both columns on the same scale
#    (otherwise rupees, in the hundreds, would outweigh orders, in single digits)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Ask K-Means to find 4 groups of similar customers
model = KMeans(n_clusters=4, random_state=42, n_init=10)
data["segment"] = model.fit_predict(X_scaled)

# 4. Describe each group by its average customer
summary = data.groupby("segment")[["orders_per_month", "avg_order_value"]].mean().round(1)
summary["customers"] = data["segment"].value_counts()

# 5. The web page
st.title("Who are my customers?")

fig, ax = plt.subplots()
ax.scatter(data["orders_per_month"], data["avg_order_value"], c=data["segment"], cmap="tab10")
ax.set_xlabel("Orders per month")
ax.set_ylabel("Average order value (Rs)")
st.pyplot(fig)

st.subheader("The 4 segments")
st.dataframe(summary)

st.subheader("Which segment is a new customer in?")
orders = st.number_input("Orders per month", 0.5, 30.0, 4.0)
value = st.number_input("Average order value (Rs)", 100, 2000, 300)

new = scaler.transform(pd.DataFrame([[orders, value]], columns=X.columns))
st.write(f"This customer belongs to **segment {model.predict(new)[0]}**.")
