import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------------
# PAGE SETUP
# --------------------------------
st.set_page_config(page_title="Smart Skincare Product Recommender", page_icon="💧", layout="wide")
st.title("💅 Smart Skincare Product Recommendation System")

st.write("""
Discover skincare products and ingredients that go well together — powered by **Apriori frequent itemset mining**.  
Choose your **skin type** and **current ingredients**, and we’ll suggest **compatible products and brands**.
""")

# --------------------------------
# LOAD DATASET
# --------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cosmetics.csv")

    # auto-detect ingredients column
    possible_cols = [c for c in df.columns if 'ingredient' in c.lower()]
    if possible_cols:
        df['ingredients'] = df[possible_cols[0]].astype(str)
    else:
        st.error("❌ No 'ingredients' column found in dataset.")
        st.stop()

    # detect brand and product name columns if present
    brand_cols = [c for c in df.columns if 'brand' in c.lower()]
    name_cols = [c for c in df.columns if 'name' in c.lower() or 'product' in c.lower()]

    df['brand'] = df[brand_cols[0]] if brand_cols else "Unknown Brand"
    df['product'] = df[name_cols[0]] if name_cols else "Unnamed Product"

    return df

data = load_data()

# --------------------------------
# DATA PROCESSING
# --------------------------------
transactions = data['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',') if i.strip() != ""])
all_ingredients = sorted(set(i for sublist in transactions for i in sublist))

# create one-hot encoded matrix
oht = pd.DataFrame(0, index=range(len(transactions)), columns=all_ingredients)
for idx, items in enumerate(transactions):
    for item in set(items):
        if item in oht.columns:
            oht.at[idx, item] = 1

# --------------------------------
# SIDEBAR INPUTS
# --------------------------------
st.sidebar.header("🧴 Your Skin Profile")

skin_types = ["Oily", "Dry", "Combination", "Sensitive", "Normal"]
skin_type = st.sidebar.selectbox("Select your skin type", skin_types)

selected_ingredients = st.sidebar.multiselect(
    "Select ingredients you currently use",
    options=all_ingredients,
    help="Pick the ingredients from your current routine."
)

min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)

# --------------------------------
# FREQUENT ITEMSET MINING
# --------------------------------
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

# --------------------------------
# PRODUCT RECOMMENDATION LOGIC
# --------------------------------
st.subheader("✨ Personalized Product Recommendations")

if selected_ingredients:
    matched_rules = rules[rules['antecedents'].apply(lambda x: all(i in x for i in selected_ingredients))]

    if not matched_rules.empty:
        recommended_ingredients = set()
        for cons in matched_rules['consequents']:
            for ing in cons.split(','):
                recommended_ingredients.add(ing.strip())

        st.success(f"🌿 Based on your ingredients and {skin_type} skin, we recommend these additional ingredients:")
        st.write(", ".join(sorted(recommended_ingredients)))

        # find matching products from dataset
        mask = data['ingredients'].apply(lambda x: any(i in x.lower() for i in recommended_ingredients))
        suggested_products = data[mask][['brand', 'product', 'ingredients']].drop_duplicates().head(10)

        st.subheader("🧴 Recommended Products")
        if not suggested_products.empty:
            st.dataframe(suggested_products)
        else:
            st.warning("No specific products found with those ingredient combinations.")
    else:
        st.warning("No matching rules for your chosen ingredients. Try adding more or lowering thresholds.")
else:
    st.subheader("💡 Top Frequent Ingredient Pairs in All Products")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).head(10))

# --------------------------------
# DOWNLOAD OPTION
# --------------------------------
csv = rules.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download All Rules (CSV)", csv, "skincare_rules.csv", "text/csv")
