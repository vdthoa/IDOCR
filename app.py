import streamlit as st
import requests
import json
import openai
import re
import tempfile
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

# Khởi tạo OpenAI client
openai.api_key = OPENAI_API_KEY
client = openai

def ocr_space_file(filename, overlay=False, api_key=OCR_SPACE_API_KEY, language='eng'):
    """ OCR.space API request with local file. """
    payload = {
        'isOverlayRequired': overlay,
        'apikey': api_key,
        'language': language,
        'detectOrientation': 'true'
    }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload)
    return r.content.decode()

def fix_json_string(json_string):
    """ Sửa JSON không hợp lệ (dấu nháy đơn, định dạng sai). """
    json_string = json_string.replace("'", '"')
    json_string = re.sub(r'\s+', ' ', json_string.strip())
    return json_string

def parse_ocr_to_json(ocr_text):
    """ Phân tích văn bản OCR thành JSON cho giấy tờ Việt Nam. """
    prompt = f"""
    Phân tích văn bản OCR từ giấy tờ tùy thân Việt Nam, sửa lỗi ký tự tiếng Việt và địa danh, trả về JSON hợp lệ:

    ```json
    {{
      "success": true,
      "document_type": "identity_card",
      "data": {{
        "personal_identification_number": null,
        "full_name": null,
        "date_of_birth": null,
        "sex": null,
        "nationality": "Việt Nam",
        "place_of_residence": null,
        "place_of_birth": null,
        "date_of_issue": null,
        "date_of_expiry": null
      }}
    }}
    ```

    Văn bản OCR:
    ```
    {ocr_text}
    ```

    Hướng dẫn:
    - Sửa lỗi OCR (ví dụ: "Hång Lia" → "Háng Lìa", "Dién Bién Döng" → "Điện Biên Đông") dựa trên địa danh Việt Nam.
    - Lấy địa chỉ đầy đủ (xã, huyện, tỉnh) cho `place_of_residence` và `place_of_birth`.
    - Nếu thông tin không rõ, để null.
    - Trả về JSON trong khối ```json ... ```, dùng dấu nháy kép cho chuỗi.
    - Nếu lỗi, trả về {{"success": false, "error": "lý do"}}.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI xử lý OCR giấy tờ Việt Nam, trả về JSON hợp lệ."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )
        response_text = response.choices[0].message.content.strip()
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if not json_match:
            return {"success": False, "error": "No JSON block found in response"}
        json_string = json_match.group(1).strip()
        json_string = fix_json_string(json_string)
        try:
            json_output = json.loads(json_string)
            return json_output
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"API error: {str(e)}"}

def merge_ocr_results(front_result, back_result):
    """ Gộp kết quả OCR từ mặt trước và mặt sau. """
    if not front_result.get("success") or not back_result.get("success"):
        return {"success": False, "error": "One or both OCR processes failed"}
    
    merged_data = {
        "success": True,
        "document_type": "identity_card",
        "data": {
            "personal_identification_number": front_result["data"].get("personal_identification_number"),
            "full_name": front_result["data"].get("full_name"),
            "date_of_birth": front_result["data"].get("date_of_birth"),
            "sex": front_result["data"].get("sex"),
            "nationality": front_result["data"].get("nationality", "Việt Nam"),
            "place_of_residence": back_result["data"].get("place_of_residence"),
            "place_of_birth": back_result["data"].get("place_of_birth"),
            "date_of_issue": back_result["data"].get("date_of_issue"),
            "date_of_expiry": back_result["data"].get("date_of_expiry")
        }
    }
    return merged_data

# Streamlit app
st.title("Vietnamese ID Card OCR Processor")
st.write("Upload cả hai mặt của thẻ CCCD để trích xuất thông tin.")

# Tạo thư mục tạm riêng cho mỗi session
session_id = str(uuid.uuid4())
temp_dir = os.path.join(tempfile.gettempdir(), session_id)
os.makedirs(temp_dir, exist_ok=True)

# File uploaders
front_image = st.file_uploader("Upload mặt trước CCCD", type=["jpg", "png"])
back_image = st.file_uploader("Upload mặt sau CCCD", type=["jpg", "png"])

if front_image and back_image:
    # Tạo tên file tạm duy nhất với UUID
    front_path = os.path.join(temp_dir, f"front_{uuid.uuid4()}.jpg")
    back_path = os.path.join(temp_dir, f"back_{uuid.uuid4()}.jpg")

    try:
        # Lưu hình ảnh vào file tạm
        with open(front_path, "wb") as f:
            f.write(front_image.read())
        with open(back_path, "wb") as f:
            f.write(back_image.read())

        # Xử lý OCR cho cả hai hình
        with st.spinner("Đang xử lý mặt trước..."):
            front_ocr = ocr_space_file(filename=front_path)
            front_result = parse_ocr_to_json(front_ocr)
        
        with st.spinner("Đang xử lý mặt sau..."):
            back_ocr = ocr_space_file(filename=back_path)
            back_result = parse_ocr_to_json(back_ocr)

        # Gộp kết quả
        final_result = merge_ocr_results(front_result, back_result)

        # Hiển thị kết quả
        st.subheader("Thông tin trích xuất")
        st.json(final_result)

    except Exception as e:
        st.error(f"Lỗi khi xử lý hình ảnh: {str(e)}")
    
    finally:
        # Xóa file tạm và thư mục tạm
        for path in [front_path, back_path]:
            if os.path.exists(path):
                os.unlink(path)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)  # Chỉ xóa thư mục nếu rỗng
            except OSError:
                pass  # Bỏ qua nếu thư mục không rỗng
else:
    st.info("Vui lòng upload cả hai hình ảnh mặt trước và mặt sau của CCCD.")