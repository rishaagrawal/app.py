import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(page_title="Skincare Ingredient Miner", page_icon="💄", layout="wide")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cosmetics.csv")
    df = df.dropna(subset=['Ingredients'])
    df['Ingredients'] = (
        df['Ingredients']
        .str.lower()
        .str.replace('[^a-z, ]', '', regex=True)
        .str.split(', ')
    )
    return df

df = load_data()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.title("⚙️ Controls")
skin_type = st.sidebar.selectbox("Select Skin Type:", ["All", "Dry", "Oily", "Sensitive"])
user_ingredient = st.sidebar.text_input("Ingredient you use (optional):", "hyaluronic acid")
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.3)
max_items = st.sidebar.slider("Max Rules to Show", 5, 50, 10)

st.title("💄 Skincare Ingredient Pattern Miner")
st.markdown(
    "Explore frequent ingredient combinations and discover what works best together for your skin!"
)

# -------------------------------
# Filter by Skin Type
# -------------------------------
if skin_type != "All" and "Label" in df.columns:
    df_filtered = df[df["Label"].str.contains(skin_type, case=False, na=False)]
else:
    df_filtered = df

# -------------------------------
# Transaction Encoding
# -------------------------------
te = TransactionEncoder()
te_ary = te.fit(df_filtered["Ingredients"]).transform(df_filtered["Ingredients"])
data = pd.DataFrame(te_ary, columns=te.columns_)

# -------------------------------
# Run Apriori and Rules Safely
# -------------------------------
try:
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        st.warning("⚠️ No frequent itemsets found. Try lowering the minimum support value.")
        rules = pd.DataFrame()
    else:
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )

        if rules.empty:
            st.warning("⚠️ No association rules found. Try lowering the confidence value.")
except Exception as e:
    st.error(f"An error occurred while generating rules: {e}")
    frequent_itemsets = pd.DataFrame()
    rules = pd.DataFrame()

# -------------------------------
# Filter Rules by Ingredient (optional)
# -------------------------------
if not rules.empty and user_ingredient:
    rules = rules[rules["antecedents"].apply(lambda x: user_ingredient in str(x))]

# -------------------------------
# Display Results
# -------------------------------
st.subheader("📊 Frequent Ingredient Sets")
if not frequent_itemsets.empty:
    st.dataframe(frequent_itemsets.head(max_items))
else:
    st.info("No frequent itemsets to display.")

st.subheader("💡 Strong Association Rules")
if not rules.empty:
    st.dataframe(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(max_items)
    )
else:
    st.info("No rules to display. Adjust support or confidence.")

# -------------------------------
# Visualization
# -------------------------------
if not frequent_itemsets.empty:
    st.subheader("🔍 Top Ingredients by Support")
    st.bar_chart(
        frequent_itemsets.nlargest(10, "support").set_index("itemsets")["support"]
    )
