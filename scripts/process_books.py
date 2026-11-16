import os
import json
import glob
import asyncio
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import whisper
from pydub import AudioSegment
import edge_tts
import google.generativeai as genai
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Cấu hình API từ GitHub Secrets
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDS_JSON")
SHEET_ID = os.environ.get("SHEET_ID")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")

genai.configure(api_key=GEMINI_API_KEY)

# Kết nối Google Sheets & Drive
creds_dict = json.loads(GOOGLE_CREDS_JSON)
scopes = [
    "https://googleapis.com",
    "https://googleapis.com"
]
creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1
drive_service = build('drive', 'v3', credentials=creds)

LOG_FILE = "processed_log.json"

def load_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_log(log_data):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

def extract_text_from_pdf(pdf_path):
    """Tự động nhận diện PDF chữ hoặc PDF scan để trích xuất văn bản."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    if len(text.strip()) < 50:
        print(f"Phát hiện file scan: {pdf_path}. Đang chạy OCR...")
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang='eng') + "\n"
    return text

def upload_audio_to_drive(file_path, file_name):
    """Tải file audio lên Google Drive và trả về link nghe trực tiếp."""
    file_metadata = {
        'name': file_name,
        'parents': [DRIVE_FOLDER_ID]
    }
    media = MediaFileUpload(file_path, mimetype='audio/mp3')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    
    drive_service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    return f"https://google.com{file_id}"

async def generate_ai_audio(text, output_path, voice="en-US-AriaNeural"):
    """Tạo giọng đọc AI tự nhiên bằng Edge-TTS."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def process_with_audio(pdf_path, audio_path, folder_name):
    """Xử lý khi có cả PDF và file Audio gốc."""
    print(f"[{folder_name}] Chạy chế độ: PDF + Audio gốc")
    raw_text = extract_text_from_pdf(pdf_path)
    
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    segments = result["segments"]
    
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    segments_text = "\n".join([f"[{s['start']:.2f} - {s['end']:.2f}]: {s['text']}" for s in segments])
    prompt = f"""
    Đối chiếu văn bản sách PDF và các đoạn thoại từ Audio để sửa lỗi chính tả, sau đó trả về danh sách JSON các flashcard.
    Văn bản sách PDF: {raw_text[:4000]}
    Các đoạn Audio: {segments_text}
    
    Yêu cầu trả về ONLY JSON array (không markdown):
    [
      {{
        "start": 0.0,
        "end": 3.5,
        "english": "Câu tiếng Anh chuẩn xác",
        "vietnamese": "Nghĩa tiếng Việt tự nhiên",
        "vocab": "Từ vựng đáng chú ý (loại từ): nghĩa"
      }}
    ]
    """
    response = gemini_model.generate_content(prompt)
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    flashcards = json.loads(clean_json)
    
    full_audio = AudioSegment.from_file(audio_path)
    os.makedirs("temp_audio", exist_ok=True)
    
    rows_to_insert = []
    for idx, card in enumerate(flashcards):
        start_ms = max(0, int(card["start"] * 1000) - 200)
        end_ms = min(len(full_audio), int(card["end"] * 1000) + 300)
        
        chunk = full_audio[start_ms:end_ms]
        chunk_filename = f"{folder_name}_card_{idx+1}.mp3"
        chunk_path = os.path.join("temp_audio", chunk_filename)
        chunk.export(chunk_path, format="mp3")
        
        audio_url = upload_audio_to_drive(chunk_path, chunk_filename)
        rows_to_insert.append([folder_name, card["english"], card["vietnamese"], card["vocab"], audio_url])
        
    if rows_to_insert:
        sheet.append_rows(rows_to_insert)
        print(f"Đã thêm {len(rows_to_insert)} flashcard (Audio gốc) vào Google Sheets!")

def process_pdf_only(pdf_path, folder_name):
    """Xử lý khi chỉ có file PDF -> Dùng giọng đọc AI."""
    print(f"[{folder_name}] Chạy chế độ: Chỉ có PDF + Giọng đọc AI")
    raw_text = extract_text_from_pdf(pdf_path)
    
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Dưới đây là nội dung trích xuất từ một sách học tiếng Anh. Hãy lọc ra các từ vựng, cụm từ hoặc câu giao tiếp quan trọng nhất trong bài để tạo thành bộ flashcard học tiếng Anh (tối đa 30 thẻ).
    
    Nội dung sách:
    {raw_text[:6000]}
    
    Yêu cầu trả về ONLY JSON array (không markdown):
    [
      {{
        "english": "Từ vựng hoặc câu tiếng Anh chuẩn xác",
        "vietnamese": "Nghĩa tiếng Việt tự nhiên",
        "vocab": "Giải thích ngữ pháp / từ loại / ví dụ ngắn"
      }}
    ]
    """
    response = gemini_model.generate_content(prompt)
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    flashcards = json.loads(clean_json)
    
    os.makedirs("temp_audio", exist_ok=True)
    rows_to_insert = []
    
    for idx, card in enumerate(flashcards):
        chunk_filename = f"{folder_name}_ai_{idx+1}.mp3"
        chunk_path = os.path.join("temp_audio", chunk_filename)
        
        # Tạo file âm thanh bằng giọng đọc AI
        asyncio.run(generate_ai_audio(card["english"], chunk_path))
        
        audio_url = upload_audio_to_drive(chunk_path, chunk_filename)
        rows_to_insert.append([folder_name, card["english"], card["vietnamese"], card["vocab"], audio_url])
        
    if rows_to_insert:
        sheet.append_rows(rows_to_insert)
        print(f"Đã thêm {len(rows_to_insert)} flashcard (Giọng đọc AI) vào Google Sheets!")

def main():
    processed = load_log()
    if not os.path.exists("books"):
        return

    for folder in os.listdir("books"):
        folder_path = os.path.join("books", folder)
        if not os.path.isdir(folder_path) or folder in processed:
            continue
            
        pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
        audios = glob.glob(os.path.join(folder_path, "*.mp3")) + glob.glob(os.path.join(folder_path, "*.m4a")) + glob.glob(os.path.join(folder_path, "*.wav"))
        
        if pdfs and audios:
            process_with_audio(pdfs[0], audios[0], folder)
            processed.append(folder)
            save_log(processed)
        elif pdfs and not audios:
            process_pdf_only(pdfs[0], folder)
            processed.append(folder)
            save_log(processed)

if __name__ == "__main__":
    main()
