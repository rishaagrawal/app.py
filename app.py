import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and dataset
frequent_itemsets, rules, df = pickle.load(open('skincare_model.pkl', 'rb'))

st.title("💆‍♀️ Skincare Product Recommendation System")

# Sidebar Filters
skin_types = ['Dry', 'Oily', 'Combination', 'Normal', 'Sensitive']
selected_skin = st.sidebar.multiselect("Select Skin Type", skin_types)

search_brand = st.sidebar.text_input("Search by Brand Name")
search_product = st.sidebar.text_input("Search by Product Name")

# Display dataset based on filters
filtered_df = df.copy()
if selected_skin:
    for skin in selected_skin:
        filtered_df = filtered_df[filtered_df[skin] == 1]
if search_brand:
    filtered_df = filtered_df[filtered_df['Brand'].str.contains(search_brand, case=False, na=False)]
if search_product:
    filtered_df = filtered_df[filtered_df['Label'].str.contains(search_product, case=False, na=False)]

st.subheader("Filtered Products")
st.dataframe(filtered_df[['Brand', 'Label', 'Ingredients']].head(20))

# Select ingredients
all_ingredients = sorted(set(i for lst in df['Ingredients'] for i in lst))
selected_ing = st.multiselect("Select Ingredients You Currently Use", all_ingredients)

if selected_ing:
    st.subheader("Frequent Itemsets")
    st.dataframe(frequent_itemsets.head(10))

    st.subheader("Top Association Rules")
    st.dataframe(rules.head(10))

    # Plot
    st.subheader("Top Frequent Itemsets (Bar Chart)")
    top_items = frequent_itemsets.sort_values('support', ascending=False).head(10)
    plt.barh(top_items['itemsets'].astype(str), top_items['support'])
    plt.xlabel('Support')
    st.pyplot(plt)

    # Recommendation section
    st.subheader("🧴 Recommended Products")
    recs = []
    for _, row in rules.iterrows():
        if set(selected_ing).issubset(row['antecedents']):
            recs.extend(list(row['consequents']))
    recs = list(set(recs))
    if recs:
        st.write(f"Products containing {', '.join(recs)}:")
        for r in recs:
            matched = df[df['Ingredients'].apply(lambda x: r in x)]
            for _, row in matched.iterrows():
                st.write(f"- **{row['Brand']}** — {row['Label']}")
    else:
        st.write("No recommendations found for the selected ingredients.")
