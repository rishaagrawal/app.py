import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------
# Title and description
# ----------------------------------
st.title("💅 Skincare Ingredient Pattern Miner")
st.write("""
Explore hidden ingredient combinations in Indian skincare products using **Apriori Algorithm**.  
Adjust thresholds to find patterns between ingredients and how they co-occur.
""")

# ----------------------------------
# Load dataset (backend)
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cosmetics.csv")
    df['ingredients'] = df['ingredients'].astype(str)
    df['ingredients'] = df['ingredients'].apply(lambda x: x.lower())
    return df

try:
    data = load_data()
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

st.subheader("📄 Dataset Preview")
st.dataframe(data.head())

# ----------------------------------
# Parameters (user input)
# ----------------------------------
st.sidebar.header("Adjust Mining Parameters")
min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)

# ----------------------------------
# Preprocess ingredients column
# ----------------------------------
transactions = data['ingredients'].apply(lambda x: [i.strip() for i in x.split(',') if i.strip() != ""])
all_items = sorted(set(i for sublist in transactions for i in sublist))
oht = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)

for idx, items in enumerate(transactions):
    oht.loc[idx, items] = 1

# ----------------------------------
# Apriori and Association Rules
# ----------------------------------
with st.spinner("⏳ Mining frequent itemsets..."):
    frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

if frequent_itemsets.empty:
    st.warning("⚠️ No frequent itemsets found. Try lowering the support threshold.")
else:
    st.subheader("📊 Frequent Ingredient Sets")
    st.dataframe(frequent_itemsets.head(10))

    with st.spinner("🔍 Generating association rules..."):
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        if not rules.empty:
            rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            rules = rules.sort_values(by="lift", ascending=False)

            st.subheader("🔗 Discovered Ingredient Associations")
            st.dataframe(rules.head(10))

            csv = rules.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Rules as CSV",
                data=csv,
                file_name='association_rules.csv',
                mime='text/csv'
            )
        else:
            st.warning("⚠️ No association rules found. Try lowering the confidence threshold.")
