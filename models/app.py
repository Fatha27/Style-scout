import streamlit as st
import requests
from PIL import Image
import os
import pandas as pd

# Set up the paths for product images and the product dataset
IMAGE_FOLDER = r'path_to_apparel_images'
CSV_PATH = r'path_to_csv'

fashion = pd.read_csv(CSV_PATH)

# Function to get the product title based on the product ID
def get_product_title(product_id):
    row = fashion[fashion['ProductId'] == int(product_id)]
    if not row.empty:
        return row.iloc[0]['ProductTitle']
    return "Title not found"

# Function to get image paths for the top similar products, ensuring unique titles
def get_unique_image_paths(product_ids):
    image_paths = []
    displayed_titles = set()

    for product_id in product_ids:
        # Get the product title from the CSV
        product_title = get_product_title(product_id)

        # Skip if the product title is a duplicate
        if product_title in displayed_titles:
            continue

        # Add to the set of displayed titles
        displayed_titles.add(product_title)

        # Map product ID to image filename (assuming the image names are in format like '37378.jpg')
        image_path = os.path.join(IMAGE_FOLDER, f"{product_id}.jpg")
        if os.path.exists(image_path):
            image_paths.append((image_path, product_title))
        else:
            st.error(f"Image for Product Title '{product_title}' not found!")

        # Stop once we have 6 unique images
        if len(image_paths) >= 6:
            break
    return image_paths

# Function to call the FastAPI service
def find_similar_products(image):
    url = "http://127.0.0.1:8000/find_similar/"  # FastAPI endpoint
    files = {"file": image}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        data = response.json()
        return data["top_6_similar_products"]
    else:
        st.error("Error retrieving similar products.")
        return []

# Streamlit UI
st.title("Style Scout")
st.write("Upload a product image to find similar products.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image with reduced size
    st.image(uploaded_image, caption="Uploaded Product Image", width=300)  
    
    # Send the uploaded image to FastAPI for similarity search
    st.write("Finding similar products...")
    similar_product_ids = find_similar_products(uploaded_image)
    
    if similar_product_ids:
        st.write("Top similar products (excluding duplicates):")
        
        # Retrieve and display the unique product images
        image_paths_titles = get_unique_image_paths(similar_product_ids)

        # Display images in 2 rows of 3 images
        cols = st.columns(3)
        for idx, (image_path, product_title) in enumerate(image_paths_titles):
            with cols[idx % 3]:
                st.image(Image.open(image_path), caption=f"Product Title: {product_title}", use_column_width=True)
