import streamlit as st
import pandas as pd
import pickle
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# Load preprocessed data or rules
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cosmetics.csv")
    df = df.dropna(subset=['Ingredients'])
    df['Ingredients'] = df['Ingredients'].str.lower().str.replace('[^a-z, ]','',regex=True).str.split(', ')
    return df

df = load_data()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💄 Skincare Ingredient Pattern Miner")
st.markdown("Find frequent ingredient combinations and get recommendations!")

skin_type = st.selectbox("Select your skin type:", ["All", "Dry", "Oily", "Sensitive"])
user_ingredient = st.text_input("Enter an ingredient you use (optional):", "hyaluronic acid")
min_support = st.slider("Minimum Support:", 0.01, 0.5, 0.05)
min_confidence = st.slider("Minimum Confidence:", 0.1, 1.0, 0.3)
max_items = st.slider("Number of itemsets to display:", 5, 50, 10)

# -------------------------------
# Prepare transactions
# -------------------------------
if skin_type != "All" and "Label" in df.columns:
    df_filtered = df[df["Label"].str.contains(skin_type, case=False, na=False)]
else:
    df_filtered = df

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(df_filtered['Ingredients']).transform(df_filtered['Ingredients'])
data = pd.DataFrame(te_ary, columns=te.columns_)

# -------------------------------
# Run Apriori dynamically
# -------------------------------
frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Filter rules containing user ingredient
if user_ingredient:
    rules = rules[rules['antecedents'].apply(lambda x: user_ingredient in str(x))]

# -------------------------------
# Display results
# -------------------------------
st.subheader("📊 Frequent Ingredient Sets")
st.dataframe(frequent_itemsets.head(max_items))

st.subheader("💡 Strong Association Rules")
if not rules.empty:
    st.dataframe(rules[['antecedents','consequents','support','confidence','lift']].head(max_items))
else:
    st.write("No strong rules found. Try lowering support/confidence.")

# -------------------------------
# Visualization
# -------------------------------
st.bar_chart(frequent_itemsets.nlargest(10, 'support').set_index('itemsets')['support'])
