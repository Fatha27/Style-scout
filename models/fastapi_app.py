import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights  

app = FastAPI()

# Simulate your extracted features and product indexes 
extracted_features = np.load('extracted_features.npy')  # Features array
product_indexes = np.load('product_indexes.npy')  # Corresponding product IDs (e.g., '24908.jpg')

# Loading pre-trained ResNet model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the top layer (classifier)
model.eval()

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the model's expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ResNet
])

@app.post("/find_similar/")
async def find_similar(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Extract features using the model
        with torch.no_grad():
            features = model(input_tensor).squeeze().numpy()  # Remove extra dimensions

        # Calculate cosine similarities between the uploaded image features and all extracted features
        similarities = cosine_similarity(features.reshape(1, -1), extracted_features)[0]
        
        # Sort similarities and get indices of top similar products (excluding the first one - itself)
        similar_indices = np.argsort(similarities)[::-1]
        #top 13 similar products(since dataset contains some duplicates we fetch 13 and skip them, only displaying 6)
        top_similar_indices = similar_indices[1:14]  
        
        # Get the product IDs corresponding to the similar products
        top_similar_product_ids = [product_indexes[idx][:-4] for idx in top_similar_indices]  # Remove '.jpg' extension

        return {"top_6_similar_products": top_similar_product_ids}
    
    except Exception as e:
        return {"error": str(e)}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
