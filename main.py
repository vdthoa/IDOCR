from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
import json
import openai
import re
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vietnamese ID Card OCR API")

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

# Check if API keys exist
if not OPENAI_API_KEY or not OCR_SPACE_API_KEY:
    raise Exception("Thiếu OPENAI_API_KEY hoặc OCR_SPACE_API_KEY trong biến môi trường")

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY
client = openai

def ocr_space_file(file_content: bytes, filename: str, api_key: str = OCR_SPACE_API_KEY, language: str = 'eng'):
    """ OCR.space API request with file content. """
    try:
        payload = {
            'isOverlayRequired': False,
            'apikey': api_key,
            'language': language,
            'detectOrientation': 'true',
            'OCREngine':2
        }
        files = {filename: file_content}
        r = requests.post('https://api.ocr.space/parse/image', files=files, data=payload)
        r.raise_for_status()
        return r.content.decode()
    except requests.RequestException as e:
        logger.error(f"Lỗi khi gọi OCR.space API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi OCR.space API: {str(e)}")

def fix_json_string(json_string: str) -> str:
    """ Fix invalid JSON string (single quotes, malformed format). """
    json_string = json_string.replace("'", '"')
    json_string = re.sub(r'\s+', ' ', json_string.strip())
    return json_string

def parse_ocr_to_json(ocr_text: str) -> dict:
    """ Parse OCR text into JSON for Vietnamese ID documents. """
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
            model="gpt-4o",  # Use lighter model for faster processing
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
            logger.error("Không tìm thấy khối JSON trong phản hồi từ OpenAI")
            return {"success": False, "error": "Không tìm thấy khối JSON trong phản hồi"}
        json_string = json_match.group(1).strip()
        json_string = fix_json_string(json_string)
        try:
            json_output = json.loads(json_string)
            return json_output
        except json.JSONDecodeError as e:
            logger.error(f"Định dạng JSON không hợp lệ: {str(e)}")
            return {"success": False, "error": f"Định dạng JSON không hợp lệ: {str(e)}"}
    except Exception as e:
        logger.error(f"Lỗi khi gọi OpenAI API: {str(e)}")
        return {"success": False, "error": f"Lỗi API: {str(e)}"}

def merge_ocr_results(front_result: dict, back_result: dict) -> dict:
    """ Merge OCR results from front and back images. """
    if not front_result.get("success") or not back_result.get("success"):
        logger.error("Một hoặc cả hai quá trình OCR thất bại")
        return {"success": False, "error": "Một hoặc cả hai quá trình OCR thất bại"}
    
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

@app.post("/process-id-card/")
async def process_id_card(
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...)
):
    """ Process uploaded front and back ID card images and return extracted information as JSON. """
    # Validate file types
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    front_ext = os.path.splitext(front_image.filename)[1].lower()
    back_ext = os.path.splitext(back_image.filename)[1].lower()
    
    if front_ext not in allowed_extensions or back_ext not in allowed_extensions:
        logger.warning(f"Định dạng file không hợp lệ: front={front_ext}, back={back_ext}")
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file JPG hoặc PNG")

    # Validate file size (5MB limit)
    if front_image.size > 5_000_000 or back_image.size > 5_000_000:
        logger.warning("Kích thước file vượt quá 5MB")
        raise HTTPException(status_code=400, detail="Kích thước file vượt quá 5MB")

    try:
        # Read file content into memory
        front_content = await front_image.read()
        back_content = await back_image.read()

        # Process OCR for both images concurrently using ThreadPoolExecutor
        logger.info("Bắt đầu xử lý OCR cho hình ảnh")
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_event_loop()
            front_task = loop.run_in_executor(executor, ocr_space_file, front_content, front_image.filename)
            back_task = loop.run_in_executor(executor, ocr_space_file, back_content, back_image.filename)
            front_ocr, back_ocr = await asyncio.gather(front_task, back_task)
        
        front_result = parse_ocr_to_json(front_ocr)
        back_result = parse_ocr_to_json(back_ocr)

        # Merge results
        final_result = merge_ocr_results(front_result, back_result)
        logger.info("Hoàn thành xử lý OCR và gộp kết quả")
        
        return final_result

    except Exception as e:
        logger.error(f"Lỗi khi xử lý hình ảnh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")