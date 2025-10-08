import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Skincare Product Recommendation", layout="wide")
st.title("💆‍♀️ Skincare Product Recommendation System")
st.markdown("Find ingredient combinations and get personalized product recommendations!")

# --------------------------
# LOAD DATA
# --------------------------
frequent_itemsets, rules, df = pickle.load(open('skincare_model.pkl', 'rb'))

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

# User input for Apriori thresholds
min_support = st.sidebar.number_input("Minimum Support", min_value=0.01, max_value=1.0, value=0.05, step=0.01)
min_confidence = st.sidebar.number_input("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

# --------------------------
# FILTER DATA BASED ON SELECTION
# --------------------------
filtered_df = df.copy()

if selected_skin:
    for skin in selected_skin:
        filtered_df = filtered_df[filtered_df[skin] == 1]

if selected_category:
    filtered_df = filtered_df[filtered_df['Label'].isin(selected_category)]

if selected_brand != "All":
    filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]

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
# RUN MODEL BUTTON
# --------------------------
run_clicked = st.button("🚀 Run Model")

if run_clicked:
    st.subheader("🔎 Running Apriori with your parameters...")
    
    # Recreate transaction data for Apriori
    te = TransactionEncoder()
    te_array = te.fit(filtered_df['Ingredients']).transform(filtered_df['Ingredients'])
    trans_data = pd.DataFrame(te_array, columns=te.columns_)

    # Apply Apriori based on user thresholds
    frequent_itemsets_user = apriori(trans_data, min_support=min_support, use_colnames=True)
    rules_user = association_rules(frequent_itemsets_user, metric="confidence", min_threshold=min_confidence)

    if frequent_itemsets_user.empty:
        st.warning("No frequent itemsets found. Try lowering the support value.")
    else:
        st.subheader("📊 Frequent Itemsets (Top 10)")
        st.dataframe(frequent_itemsets_user.sort_values('support', ascending=False).head(10))

        st.subheader("📈 Association Rules (Top 10)")
        if rules_user.empty:
            st.warning("No rules found. Try lowering the confidence value.")
        else:
            st.dataframe(rules_user[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

            # Plot bar chart
            st.subheader("📉 Top Frequent Itemsets (Bar Chart)")
            top_items = frequent_itemsets_user.sort_values('support', ascending=False).head(10)
            plt.figure(figsize=(8, 4))
            plt.barh(top_items['itemsets'].astype(str), top_items['support'])
            plt.xlabel('Support')
            plt.ylabel('Itemsets')
            st.pyplot(plt)

    # --------------------------
    # RECOMMENDATION SECTION
    # --------------------------
    if selected_ing:
        st.subheader("💡 Recommended Products Based on Your Ingredients")
        recs = []
        for _, row in rules_user.iterrows():
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
            st.info("No recommendations found for the selected ingredients.")
    else:
        st.info("Select ingredients to get personalized recommendations.")
else:
    st.info("🧠 Adjust your filters and parameters, then click **Run Model** to generate results.")

