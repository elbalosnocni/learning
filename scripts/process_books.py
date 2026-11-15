import os
import json
import glob
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
import whisper
from pydub import AudioSegment
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
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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
    
    # Nếu ít chữ, khả năng cao là file scan -> Chuyển sang OCR
    if len(text.strip()) < 50:
        print(f"Phát hiện file scan: {pdf_path}. Đang chạy OCR...")
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang='eng') + "\n"
    return text

def upload_audio_to_drive(file_path, file_name):
    """Tải đoạn audio đã cắt lên Google Drive và trả về link nghe trực tiếp."""
    file_metadata = {
        'name': file_name,
        'parents': [DRIVE_FOLDER_ID]
    }
    media = MediaFileUpload(file_path, mimetype='audio/mp3')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    
    # Mở quyền xem công khai để flashcard có thể phát âm thanh
    drive_service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    return f"https://google.com{file_id}"

def process_pair(pdf_path, audio_path, folder_name):
    print(f"Đang xử lý bộ: {folder_name}")
    raw_text = extract_text_from_pdf(pdf_path)
    
    # 1. Dùng Whisper lấy timestamp từng câu trong file audio dài
    print("Đang phân tích mốc thời gian audio bằng Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="en")
    segments = result["segments"]
    
    # 2. Dùng Gemini để chuẩn hóa câu, dịch nghĩa và trích xuất từ vựng
    print("Đang tạo nội dung flashcard bằng Gemini...")
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Gửi các segment từ Whisper kèm ngữ cảnh từ PDF để Gemini đối chiếu và dịch
    segments_text = "\n".join([f"[{s['start']:.2f} - {s['end']:.2f}]: {s['text']}" for s in segments])
    prompt = f"""
    Dưới đây là văn bản gốc từ sách PDF và các đoạn thoại đã nhận diện thời gian từ file Audio.
    Hãy đối chiếu để sửa lỗi chính tả (nếu có) từ bản nhận diện audio dựa theo sách PDF, sau đó trả về danh sách JSON các flashcard.
    
    Văn bản sách PDF:
    {raw_text[:4000]}
    
    Các đoạn Audio:
    {segments_text}
    
    Yêu cầu định dạng trả về ONLY JSON array (không markdown):
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
    
    # 3. Cắt audio, upload Drive và ghi vào Google Sheets
    full_audio = AudioSegment.from_file(audio_path)
    os.makedirs("temp_audio", exist_ok=True)
    
    rows_to_insert = []
    for idx, card in enumerate(flashcards):
        start_ms = max(0, int(card["start"] * 1000) - 200) # Dư 0.2s đầu
        end_ms = min(len(full_audio), int(card["end"] * 1000) + 300) # Dư 0.3s cuối
        
        chunk = full_audio[start_ms:end_ms]
        chunk_filename = f"{folder_name}_card_{idx+1}.mp3"
        chunk_path = os.path.join("temp_audio", chunk_filename)
        chunk.export(chunk_path, format="mp3")
        
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
        print(f"Đã thêm {len(rows_to_insert)} flashcard vào Google Sheets!")

def main():
    processed = load_log()
    # Quét tất cả thư mục con trong books/
    for folder in os.listdir("books"):
        folder_path = os.path.join("books", folder)
        if not os.path.isdir(folder_path) or folder in processed:
            continue
            
        pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
        audios = glob.glob(os.path.join(folder_path, "*.mp3")) + glob.glob(os.path.join(folder_path, "*.m4a")) + glob.glob(os.path.join(folder_path, "*.wav"))
        
        if pdfs and audios:
            process_pair(pdfs[0], audios[0], folder)
            processed.append(folder)
            save_log(processed)

if __name__ == "__main__":
    main()
