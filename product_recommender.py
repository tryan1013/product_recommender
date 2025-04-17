import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Set up the page
st.set_page_config(page_title="Fitness Gift Recommender", page_icon="🎁")

# ✅ Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("product_recommender_etsy_products.csv")
    df["description"] = df["description"].fillna("")
    df["text"] = df["title"] + " " + df["description"]
    return df

df = load_data()

# ✅ Fit TF-IDF model
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["text"])

# ✅ Streamlit UI
st.title("🎁 Fitness Gift Recommender")
st.write("Looking for the perfect gift for a fitness lover? Enter a phrase and we'll recommend a few thoughtful items from Etsy!")

user_input = st.text_input("💬 What kind of gift are you looking for? (e.g., 'funny gym mug', 'protein scoop', 'custom water bottle')")

if user_input:
    # Compute similarity scores
    input_vector = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]

    st.subheader("🎯 Top Gift Ideas")

    for idx in top_indices:
        product = df.iloc[idx]
        st.markdown(f"### [{product['title']}]({product['link']})")
        
        # Add product image
        if "image_url" in product and pd.notna(product["image_url"]):
            st.image(product["image_url"], width=300)

        st.markdown(f"- 💬 *{product['description']}*")
        st.markdown(f"- 💲 **Price:** ${product['price']}")
        st.markdown(f"- ⭐ **Rating:** {product['rating']} ({product['review_count']} reviews)")
        st.markdown("---")


    st.caption("📝 Prices and ratings are for demonstration purposes and may not reflect live Etsy listings.")
