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

app = FastAPI(title="Vietnamese ID Card and Vehicle Registration OCR API")

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
            'scale': 'true',
            'OCREngine': 2
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

def preprocess_ocr_text(ocr_text: str) -> str:
    """Tiền xử lý văn bản OCR, tập trung vào gộp Place of origin và Place of residence."""
    # Chuẩn hóa khoảng trắng và xuống dòng
    ocr_text = re.sub(r"\s*\n\s*", "\n", ocr_text.strip())
    
    # Sửa lỗi nhãn "Giới tinh" thành "Giới tính"
    ocr_text = re.sub(r"Giới tinh", "Giới tính", ocr_text, flags=re.IGNORECASE)
    
    # Gộp "Quê quán" hoặc "Place of origin" với nội dung bị tách dòng
    ocr_text = re.sub(
        r"(Quê quán|Place of origin)\s*[:\-]?\s*\n([^\n]+(?:\n[^\n]+)*)",
        lambda m: "{}: {}".format(m.group(1), re.sub(r"\n", ", ", m.group(2).strip())),
        ocr_text,
        flags=re.IGNORECASE
    )
    
    # Gộp "Nơi thường trú" hoặc "Place of residence" với nội dung bị tách dòng
    ocr_text = re.sub(
        r"(Nơi thường trú|Place of residence)\s*[:\-]?\s*\n([^\n]+(?:\n[^\n]+)*)",
        lambda m: "{}: {}".format(m.group(1), re.sub(r"\n", ", ", m.group(2).strip())),
        ocr_text,
        flags=re.IGNORECASE
    )
    
    # Loại bỏ các dòng tiêu đề không cần thiết
    ocr_text = re.sub(
        r"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n.*?\nSOCIALIST REPUBLIC OF VIET NAM\n.*?\nCĂN CƯỚC CÔNG DÂN\n.*?\nCitizen Identity Card",
        "",
        ocr_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    return ocr_text.strip()
def parse_ocr_to_json(ocr_text: str, document_type: str = "identity_card") -> dict:
    """ Parse OCR text into JSON for Vietnamese documents. """
    if document_type == "identity_card":
        ocr_text = preprocess_ocr_text(ocr_text)

        prompt = f"""
        Phân tích văn bản OCR từ giấy tờ tùy thân Việt Nam. **Sửa lỗi ký tự tiếng Việt và chuẩn hóa họ tên và địa danh về đúng tên hành chính Việt Nam**. Sau đó trích xuất các thông tin vào JSON theo định dạng sau:

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
            "place_of_origin": null,
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
        -Trích xuất các trường: mã định danh (Số/No./ID), họ tên, ngày sinh, giới tính, quốc tịch, nơi thường trú, nơi sinh, quê quán, ngày cấp, ngày hết hạn.
        -Chuẩn hóa ngày tháng từ DD/MM/YYYY hoặc DD-MM-YYYY sang YYYY-MM-DD.
        -Chuẩn hóa giới tính: "Male" → "Nam", "Female" → "Nữ".
        -Địa chỉ như Quê quán hoặc Nơi thường trú đã được gộp thành một dòng, chứa dấu phẩy giữa các phần (ví dụ: "14/20 Hoàng Diệu, Tây Lộc, Thành phố Huế, Thừa Thiên Huế").
        -Bỏ qua các dòng tiêu đề như "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CĂN CƯỚC CÔNG DÂN"…
        -Nếu phát hiện tên địa danh sai do OCR (ví dụ: "Phủ Thượng"), hãy tự động sửa về đúng tên hành chính thực tế ("Phú Thượng").
        -Có thể sử dụng kiến thức về địa danh Việt Nam để sửa lỗi như: "Dién Biên Döng" → "Điện Biên Đông", "Thùa Thiên Huế" → "Thừa Thiên Huế", "Tp. Hô Chi Minh" → "TP. Hồ Chí Minh"...
        -Nếu thông tin không rõ ràng hoặc thiếu, để giá trị là null.
        -Chỉ trả về JSON hợp lệ, không thêm bất kỳ văn bản mô tả nào khác.
        """
    elif document_type == "motorcycle":
        prompt = f"""
        Phân tích văn bản OCR từ giấy đăng ký xe máy Việt Nam, **Sửa lỗi ký tự tiếng Việt và chuẩn hóa  họ tên và địa danh về đúng tên hành chính Việt Nam**, trả về JSON hợp lệ:

        ```json
        {{
          "success": true,
          "document_type": "motorcycle",
          "data": {{
            "full_name": null,
            "address": null,
            "brand": null,
            "model_code": null,
            "engine_no": null,
            "chassis_no": null,
            "color": null,
            "plate_no": null
          }}
        }}
        ```

        Văn bản OCR:
        ```
        {ocr_text}
        ```

        Hướng dẫn:
        - Sửa lỗi OCR (ví dụ: "Hà Nôi" → "Hà Nội", "TP Hô Chí Minh" → "TP Hồ Chí Minh") dựa trên địa danh Việt Nam.
        - Lấy địa chỉ đầy đủ (xã, huyện, tỉnh) cho `address`.
        - Nếu thông tin không rõ, để null.
        - Trả về JSON trong khối ```json ... ```, dùng dấu nháy kép cho chuỗi.
        - Nếu lỗi, trả về {{"success": false, "error": "lý do"}}.
        """
    elif document_type == "car":
        prompt = f"""
        Phân tích văn bản OCR từ giấy đăng ký xe ô tô Việt Nam, **Sửa lỗi ký tự tiếng Việt và chuẩn hóa  họ tên và  địa danh về đúng tên hành chính Việt Nam**, trả về JSON hợp lệ:

        ```json
        {{
          "success": true,
          "document_type": "car",
          "data": {{
            "address": null,
            "brand": null,
            "model_code": null,
            "engine_no": null,
            "chassis_no": null,
            "color": null,
            "plate_no": null,
            "seating_capacity": null,
            "date_of_expiry": null
          }}
        }}
        ```

        Văn bản OCR:
        ```
        {ocr_text}
        ```

        Hướng dẫn:
        - Sửa lỗi OCR (ví dụ: "Hà Nôi" → "Hà Nội", "TP Hô Chí Minh" → "TP Hồ Chí Minh") dựa trên địa danh Việt Nam.
        - Lấy địa chỉ đầy đủ (xã, huyện, tỉnh) cho `address`.
        - Nếu thông tin không rõ, để null.
        - Trả về JSON trong khối ```json ... ```, dùng dấu nháy kép cho chuỗi.
        - Nếu lỗi, trả về {{"success": false, "error": "lý do"}}.
        """
    elif document_type == "car-inspection":
        prompt = f"""
        Phân tích văn bản OCR từ giấy đăng kiểm xe ô tô Việt Nam, **Sửa lỗi ký tự tiếng Việt và chuẩn hóa  họ tên và  địa danh về đúng tên hành chính Việt Nam**, trả về JSON hợp lệ:

        ```json
        {{
          "success": true,
          "document_type": "car",
          "data": {{
            "brand": null,
            "model_code": null,
            "engine_no": null,
            "chassis_no": null,
            "type": null,
            "capacity": null,
            "plate_no": null,
            "seating_capacity": null,
            "date_of_expiry": null
          }}
        }}
        ```

        Văn bản OCR:
        ```
        {ocr_text}
        ```

        Hướng dẫn:
        - Sửa lỗi OCR (ví dụ: "Hà Nôi" → "Hà Nội", "TP Hô Chí Minh" → "TP Hồ Chí Minh") dựa trên địa danh Việt Nam.
        - Lấy địa chỉ đầy đủ (xã, huyện, tỉnh) cho `address`.
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
            "place_of_residence": front_result["data"].get("place_of_residence") or back_result["data"].get("place_of_residence"),
            "place_of_birth": back_result["data"].get("place_of_birth") or front_result["data"].get("place_of_origin"),
            "date_of_issue": back_result["data"].get("date_of_issue"),
            "date_of_expiry": back_result["data"].get("date_of_expiry") or front_result["data"].get("date_of_expiry")
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
        
        front_result = parse_ocr_to_json(front_ocr, document_type="identity_card")
        back_result = parse_ocr_to_json(back_ocr, document_type="identity_card")

        # Merge results
        final_result = merge_ocr_results(front_result, back_result)
        logger.info("Hoàn thành xử lý OCR và gộp kết quả")
        
        return final_result

    except Exception as e:
        logger.error(f"Lỗi khi xử lý hình ảnh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")

@app.post("/process-motobike-registration/")
async def process_vehicle_registration(
    image: UploadFile = File(...)
):
    """ Process uploaded vehicle registration image and return extracted information as JSON. """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(image.filename)[1].lower()
    
    if ext not in allowed_extensions:
        logger.warning(f"Định dạng file không hợp lệ: {ext}")
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file JPG hoặc PNG")

    # Validate file size (5MB limit)
    if image.size > 5_000_000:
        logger.warning("Kích thước file vượt quá 5MB")
        raise HTTPException(status_code=400, detail="Kích thước file vượt quá 5MB")

    try:
        # Read file content into memory
        content = await image.read()

        # Process OCR for the image
        logger.info("Bắt đầu xử lý OCR cho giấy đăng ký xe máy")
        ocr_result = await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(max_workers=1), 
            ocr_space_file, 
            content, 
            image.filename
        )
        
        # Parse OCR result
        result = parse_ocr_to_json(ocr_result, document_type="motorcycle")
        logger.info("Hoàn thành xử lý OCR cho giấy đăng ký xe máy")
        
        return result

    except Exception as e:
        logger.error(f"Lỗi khi xử lý hình ảnh giấy đăng ký xe máy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")
    
@app.post("/process-car-registration/")
async def process_vehicle_registration(
    image: UploadFile = File(...)
):
    """ Process uploaded vehicle registration image and return extracted information as JSON. """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(image.filename)[1].lower()
    
    if ext not in allowed_extensions:
        logger.warning(f"Định dạng file không hợp lệ: {ext}")
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file JPG hoặc PNG")

    # Validate file size (5MB limit)
    if image.size > 5_000_000:
        logger.warning("Kích thước file vượt quá 5MB")
        raise HTTPException(status_code=400, detail="Kích thước file vượt quá 5MB")

    try:
        # Read file content into memory
        content = await image.read()

        # Process OCR for the image
        logger.info("Bắt đầu xử lý OCR cho giấy đăng ký ô tô")
        ocr_result = await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(max_workers=1), 
            ocr_space_file, 
            content, 
            image.filename
        )
        
        # Parse OCR result
        result = parse_ocr_to_json(ocr_result, document_type="car")
        logger.info("Hoàn thành xử lý OCR cho giấy đăng ký ô tô")
        
        return result

    except Exception as e:
        logger.error(f"Lỗi khi xử lý hình ảnh giấy đăng ký ô tô: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")
    
@app.post("/process-car-inspection/")
async def process_vehicle_registration(
    image: UploadFile = File(...)
):
    """ Process uploaded vehicle inspection image and return extracted information as JSON. """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(image.filename)[1].lower()
    
    if ext not in allowed_extensions:
        logger.warning(f"Định dạng file không hợp lệ: {ext}")
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file JPG hoặc PNG")

    # Validate file size (5MB limit)
    if image.size > 5_000_000:
        logger.warning("Kích thước file vượt quá 5MB")
        raise HTTPException(status_code=400, detail="Kích thước file vượt quá 5MB")

    try:
        # Read file content into memory
        content = await image.read()

        # Process OCR for the image
        logger.info("Bắt đầu xử lý OCR cho giấy đăng kiểm ô tô")
        ocr_result = await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(max_workers=1), 
            ocr_space_file, 
            content, 
            image.filename
        )
        
        # Parse OCR result
        result = parse_ocr_to_json(ocr_result, document_type="car-inspection")
        logger.info("Hoàn thành xử lý OCR cho giấy đăng kiểm ô tô")
        
        return result

    except Exception as e:
        logger.error(f"Lỗi khi xử lý hình ảnh giấy đăng kiểm ô tô: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")