import os
import json
import glob
import time
import asyncio
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import edge_tts
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")
SHEET_ID = os.environ.get("SHEET_ID")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")

genai.configure(api_key=GEMINI_API_KEY)

creds_dict = json.loads(GOOGLE_CREDS_JSON)
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1
drive_service = build('drive', 'v3', credentials=creds)

LOG_FILE = "processed_log.json"
PAGES_PER_BATCH = 3  # Xử lý mỗi lần 3 trang để không chạm giới hạn API

def load_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_log(log_data):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def extract_text_by_pages(pdf_path, start_page, end_page):
    """Trích xuất văn bản theo cụm trang, tự động OCR nếu là file scan."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    end_page = min(end_page, total_pages)
    
    text = ""
    for p in range(start_page, end_page):
        text += doc[p].get_text()
        
    if len(text.strip()) < 30:
        print(f"Phát hiện trang scan: Chạy OCR từ trang {start_page+1} đến {end_page}...")
        images = convert_from_path(pdf_path, first_page=start_page+1, last_page=end_page)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang='eng') + "\n"
            
    return text, total_pages

def upload_audio_to_drive(file_path, file_name):
    file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype='audio/mp3')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    
    drive_service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    return f"https://google.com{file_id}"

async def generate_ai_audio(text, output_path, voice="en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def process_pdf_book(pdf_path, folder_name, processed_log):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    gemini_model = genai.GenerativeModel('gemini-3.8-flash')
    os.makedirs("temp_audio", exist_ok=True)

    for start_p in range(0, total_pages, PAGES_PER_BATCH):
        end_p = min(start_p + PAGES_PER_BATCH, total_pages)
        batch_id = f"{folder_name}_p{start_p+1}-{end_p}"
        
        if batch_id in processed_log:
            continue

        print(f"Đang xử lý {batch_id} / tổng {total_pages} trang...")
        raw_text, _ = extract_text_by_pages(pdf_path, start_p, end_p)
        
        if len(raw_text.strip()) < 10:
            processed_log.append(batch_id)
            save_log(processed_log)
            continue

        prompt = f"""
        Lọc toàn bộ từ vựng, cụm từ hoặc câu giao tiếp tiếng Anh quan trọng có trong đoạn sách sau để tạo flashcard.
        Nội dung sách:
        {raw_text}
        
        Yêu cầu trả về ONLY JSON array (không markdown):
        [
          {{
            "english": "Từ hoặc câu tiếng Anh chuẩn xác",
            "vietnamese": "Nghĩa tiếng Việt tự nhiên",
            "vocab": "Từ loại / Phiên âm IPA / Ví dụ ngắn"
          }}
        ]
        """
        try:
            response = gemini_model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            flashcards = json.loads(clean_json)
            
            rows_to_insert = []
            for idx, card in enumerate(flashcards):
                chunk_filename = f"{batch_id}_card_{idx+1}.mp3"
                chunk_path = os.path.join("temp_audio", chunk_filename)
                
                asyncio.run(generate_ai_audio(card["english"], chunk_path))
                audio_url = upload_audio_to_drive(chunk_path, chunk_filename)
                
                rows_to_insert.append([
                    folder_name,
                    card["english"],
                    card["vietnamese"],
                    card["vocab"],
                    audio_url
                ])
                
            if rows_to_insert:
                sheet.append_rows(rows_to_insert)
                print(f"-> Đã lưu {len(rows_to_insert)} thẻ từ {batch_id}")
                
            processed_log.append(batch_id)
            save_log(processed_log)
            
            # Nghỉ 5 giây để tránh chạm giới hạn tốc độ của Gemini API miễn phí
            time.sleep(5)
            
        except Exception as e:
            print(f"Tạm dừng tại {batch_id} do lỗi: {e}. Lần chạy sau sẽ tự chạy tiếp.")
            break

def main():
    processed = load_log()
    if not os.path.exists("books"):
        return

    for folder in os.listdir("books"):
        folder_path = os.path.join("books", folder)
        if not os.path.isdir(folder_path):
            continue
            
        pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
        for pdf_file in pdfs:
            process_pdf_book(pdf_file, folder, processed)

if __name__ == "__main__":
    main()
