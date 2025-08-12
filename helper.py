from google.cloud import vision
import io
import re

# Đường dẫn đến file key JSON của bạn
client = vision.ImageAnnotatorClient.from_service_account_file("service_account.json")

def ocr_id_card(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return {}

    full_text = texts[0].description
    return full_text

# # Test
text = ocr_id_card("id_card.jpg")
print(text) 

