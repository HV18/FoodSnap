import torch
import modelbit
from torchvision import transforms
from PIL import Image
import io
import json
import os


modelbit.login()

def FoodSnap_Predict(image_input):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CATEGORIES_JSON_PATH = os.path.join(SCRIPT_DIR, "categories.json")


    if not os.path.exists(CATEGORIES_JSON_PATH):
        raise FileNotFoundError(f"{CATEGORIES_JSON_PATH} not found. Ensure the file exists in the directory.")
    with open(CATEGORIES_JSON_PATH, 'r') as f:
        categories = json.load(f)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model = modelbit.get_model("FoodSnapV0")
    model.eval()

    if isinstance(image_input, str):  # File path
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):  # Raw image bytes
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        raise ValueError("The input should be a file path (str) or raw image bytes (bytes).")
    
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor).logits
    
    _, predicted_class = torch.max(outputs, 1)
    prediction_prob = torch.nn.functional.softmax(outputs, dim=1)[0, predicted_class].item()
    predicted_label = categories[str(predicted_class.item())]
    
    return predicted_label, prediction_prob

modelbit.deploy(FoodSnap_Predict)