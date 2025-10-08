import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# Title
# -------------------------------
st.title("💅 Skincare Ingredient Pattern Miner")
st.write("""
Discover hidden ingredient combinations in cosmetic products using the **Apriori algorithm**.
Adjust support and confidence to explore associations between ingredients.
""")

# -------------------------------
# Upload dataset
# -------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    # Check for 'ingredients' column
    if 'ingredients' not in data.columns:
        st.error("❌ The CSV must have a column named 'ingredients'.")
    else:
        # Clean and prepare
        st.sidebar.header("Algorithm Parameters")
        min_support = st.sidebar.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01)
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5, 0.05)

        # Split ingredients into lists
        data['ingredients'] = data['ingredients'].astype(str)
        transactions = data['ingredients'].apply(lambda x: [i.strip().lower() for i in x.split(',') if i.strip() != ""])

        # Create one-hot encoded dataframe
        all_items = sorted(set(i for sublist in transactions for i in sublist))
        oht = pd.DataFrame(0, index=range(len(transactions)), columns=all_items)
        for idx, items in enumerate(transactions):
            oht.loc[idx, items] = 1

        # Run Apriori
        with st.spinner("⏳ Mining frequent itemsets..."):
            frequent_itemsets = apriori(oht, min_support=min_support, use_colnames=True)
            frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)

        if frequent_itemsets.empty:
            st.warning("⚠️ No frequent itemsets found. Try lowering the support threshold.")
        else:
            st.subheader("📊 Frequent Ingredient Sets")
            st.dataframe(frequent_itemsets.head(10))

            # Generate rules
            with st.spinner("🔍 Generating association rules..."):
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                rules = rules.sort_values(by="lift", ascending=False)

            if rules.empty:
                st.warning("⚠️ No association rules found. Try lowering the confidence threshold.")
            else:
                st.subheader("🔗 Discovered Ingredient Associations")
                st.dataframe(rules.head(10))

                # Download option
                csv = rules.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Rules as CSV",
                    data=csv,
                    file_name='association_rules.csv',
                    mime='text/csv'
                )

else:
    st.info("👈 Upload a cosmetics dataset (with an 'ingredients' column) to start.")
