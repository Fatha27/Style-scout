# Style Scout

## Description
The Similar Product Finder is an AI-driven application designed to enhance e-commerce experiences by identifying and displaying similar apparel. By leveraging deep learning techniques and image processing, this system helps users find products that closely match their preferences based on visual input.

## Features
1. **Feature Extraction**: Utilizes a pre-trained `ResNet50` model (with the top layer removed) to extract relevant features from images of apparel (dataset of ~1000 boys apparel images), ensuring effective representation of product characteristics.
2. **Similarity Calculation**: Implements `cosine similarity` to measure the similarity between product images, allowing the system to identify and rank similar items accurately.
3. **User-Friendly Interface**: Offers a `Streamlit app` that allows users to upload an image of a product and receive a list of the top 6 similar products along with the product titles and images.
4. **Efficient Backend**: Built with `FastAPI` for serving requests, ensuring rapid responses and smooth user interactions.

## Usage
1. Upload an image of a footwear or apparel item using the provided interface.
2. The system will extract features from the uploaded image and calculate the similarity with stored product images.
3. View the top 5 similar products displayed along with their names.

## Sample Results
![Screenshot 2024-10-19 211356](https://github.com/user-attachments/assets/9e4577eb-5a1f-461a-99f0-4e8e9dccc2c2)
----
![Screenshot 2024-10-19 211132](https://github.com/user-attachments/assets/6e73b9c5-11a8-4ff7-b647-4db9a89c9e5f)
