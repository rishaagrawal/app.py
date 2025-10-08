import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the saved model and dataset
frequent_itemsets, rules, df = pickle.load(open('skincare_model.pkl', 'rb'))

# --------------------------
# PAGE TITLE & DESCRIPTION
# --------------------------
st.set_page_config(page_title="Skincare Product Recommendation", layout="wide")
st.title("💆‍♀️ Skincare Product Recommendation System")
st.markdown("Discover products and ingredients that match your skin type and preferences.")

# --------------------------
# SIDEBAR FILTERS
# --------------------------
st.sidebar.header("🔍 Filters")

# Skin type selection
skin_types = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
selected_skin = st.sidebar.multiselect("Select Skin Type(s)", skin_types)

# Product category filter (Label column)
product_categories = sorted(df['Label'].unique().tolist())
selected_category = st.sidebar.multiselect("Select Product Category", product_categories)

# Brand dropdown
brand_list = sorted(df['Brand'].unique().tolist())
selected_brand = st.sidebar.selectbox("Select Brand (optional)", ["All"] + brand_list)

# Product dropdown
product_list = sorted(df['Name'].unique().tolist())
selected_product = st.sidebar.selectbox("Select Product (optional)", ["All"] + product_list)

# --------------------------
# FILTERING THE DATA
# --------------------------
filtered_df = df.copy()

# Filter by skin type
if selected_skin:
    for skin in selected_skin:
        filtered_df = filtered_df[filtered_df[skin] == 1]

# Filter by product category
if selected_category:
    filtered_df = filtered_df[filtered_df['Label'].isin(selected_category)]

# Filter by brand
if selected_brand != "All":
    filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]

# Filter by product
if selected_product != "All":
    filtered_df = filtered_df[filtered_df['Name'] == selected_product]

# --------------------------
# DISPLAY FILTERED PRODUCTS
# --------------------------
st.subheader("📋 Filtered Products")
if filtered_df.empty:
    st.warning("No products found for the selected filters.")
else:
    st.dataframe(filtered_df[['Brand', 'Name', 'Label', 'Price', 'Rank', 'Ingredients']].reset_index(drop=True))

# --------------------------
# INGREDIENT SELECTION
# --------------------------
all_ingredients = sorted(set(i for lst in df['Ingredients'] for i in lst))
selected_ing = st.multiselect("Select Ingredients You Currently Use", all_ingredients)

# --------------------------
# APRIORI RESULTS AND PLOTS
# --------------------------
if selected_ing:
    st.subheader("📊 Frequent Itemsets (Top 10)")
    st.dataframe(frequent_itemsets.head(10))

    st.subheader("📈 Top Association Rules (Confidence ≥ 0.6)")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    # Plot top frequent itemsets
    st.subheader("📉 Top Frequent Itemsets (Bar Chart)")
    top_items = frequent_itemsets.sort_values('support', ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    plt.barh(top_items['itemsets'].astype(str), top_items['support'])
    plt.xlabel('Support')
    plt.ylabel('Itemsets')
    st.pyplot(plt)

    # --------------------------
    # RECOMMENDATION SECTION
    # --------------------------
    st.subheader("💡 Recommended Products Based on Selected Ingredients")

    recs = []
    for _, row in rules.iterrows():
        if set(selected_ing).issubset(row['antecedents']):
            recs.extend(list(row['consequents']))
    recs = list(set(recs))

    if recs:
        st.write(f"Products containing **{', '.join(recs)}**:")
        for r in recs:
            matched = df[df['Ingredients'].apply(lambda x: r in x)]
            for _, row in matched.iterrows():
                st.markdown(f"- **{row['Brand']}** — *{row['Name']}* ({row['Label']})")
    else:
        st.info("No recommendations found for the selected ingredients. Try adding more or different ones!")
else:
    st.info("👆 Select some ingredients above to see recommendations.")

