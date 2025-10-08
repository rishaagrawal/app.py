import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="Smart Skincare Ingredient Recommender", page_icon="💧", layout="wide")
st.title("💅 Smart Skincare Ingredient Recommendation System")

st.write("""
This app analyzes skincare products and their ingredients to find **frequent co-occurrence patterns**.  
Based on your **skin type** and **current ingredients**, it recommends other compatible ingredients that commonly appear together in successful products.
""")

# -------------------------------
# LOAD DATASET (BACKEND)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cosmetics.csv")
    # auto-detect the ingredients column
    possible_cols = [c for c in df.columns if 'ingredient' in c.lower()]
    if possible_cols:
        df['ingredients'] = df[possible_cols[0]].astype(str)
    else:
        st.error("❌ Could not find an 'ingredients' column in dataset.")
        st.stop()
    return df

data = load_data()

# -------------------------------
# INGREDIENT PROCESSING
# -------------------------------
transactions = data['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',') if i.strip() != ""])
all_ingredients = sorted(set(i for sublist in transactions for i in sublist))

# Create one-hot encoded DataFrame
oht = pd.DataFrame(0, index=range(len(transactions)), columns=all_ingredients)
for idx, items in enumerate(transactions):
    for item in set(items):
        if item in oht.columns:
            oht.at[idx, item] = 1

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("🧴 Your Skin Profile")

# Skin type input
skin_types = ["Oily", "Dry", "Combination", "Sensitive", "Normal"]
skin_type = st.sidebar.selectbox("Select your skin type", skin_types)

# Ingredient selection
selected_ingredients = st.sidebar.multiselect(
    "Select ingredients you currently use",
    options=all_ingredients,
    help="Choose the ingredients from your current routine."
)

min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)

# -------------------------------
# RUN APRIORI
# -------------------------------
st.info("⏳ Mining frequent ingredient patterns... please wait.")

frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
if frequent_itemsets.empty:
    st.error("No frequent itemsets found. Try lowering support.")
    st.stop()

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
if rules.empty:
    st.error("No association rules found. Try lowering confidence.")
    st.stop()

rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# -------------------------------
# RECOMMENDATION LOGIC
# -------------------------------
if selected_ingredients:
    matching_rules = rules[rules['antecedents'].apply(lambda x: all(i in x for i in selected_ingredients))]
    if not matching_rules.empty:
        recommendations = (
            matching_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            .sort_values(by='lift', ascending=False)
            .head(10)
        )

        st.success(f"✨ Recommended ingredients for {skin_type} skin based on your selection:")
        st.dataframe(recommendations)
    else:
        st.warning("No strong co-occurrence found for the selected ingredients. Try more ingredients or lower thresholds.")
else:
    st.subheader("💡 Frequent Ingredient Patterns Across All Products")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))

# -------------------------------
# DOWNLOAD OPTION
# -------------------------------
csv = rules.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download All Rules (CSV)", csv, "skincare_rules.csv", "text/csv")
