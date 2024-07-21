# import streamlit as st
# from PIL import Image
# import numpy as np
# import torch
# from torchvision import transforms
# from transformers import BertForSequenceClassification, BertTokenizer

# # Load the trained model
# model_name = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name)
# model.eval()

# # Define labels
# labels = ['Bacterial diseases - Aeromoniasis',
#           'Bacterial gill disease',
#           'Bacterial Red disease',
#           'Fungal diseases',
#           'Healthy Fish',
#           'Parasitic diseases',
#           'Viral diseases White tail disease']

# # Define function to preprocess image
# def preprocess_image(image):
#     image = np.array(image)
#     image = image.resize((224, 224))  # Assuming ResNet50 input shape
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0)
#     image = torch.tensor(image)
#     return image

# # Streamlit app
# def main():
#     st.title("Fish Disease Prediction")

#     # Upload image
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)

#         # Preprocess image
#         processed_image = preprocess_image(image)

#         # Predict
#         with torch.no_grad():
#             outputs = model(processed_image)
#             _, predicted_class = torch.max(outputs[0], 1)

#         st.write(f"Predicted class: {labels[predicted_class.item()]}")

# if __name__ == "__main__":
#     main()



import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:/REASEARCH_PRO 2024 SAM/fish_disease/your_model.h5')

# Define labels
labels = ['Bacterial diseases - Aeromoniasis',
          'Bacterial gill disease',
          'Bacterial Red disease',
          'Fungal diseases',
          'Healthy Fish',
          'Parasitic diseases',
          'Viral diseases White tail disease']

# Define function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Assuming ResNet50 input shape
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit app
def main():
    st.title("Fish Disease Prediction")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]

        st.write(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
