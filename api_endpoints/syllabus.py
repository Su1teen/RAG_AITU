from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import uuid
import shutil
import logging
from typing import List, Dict, Optional
import requests
import cv2
import numpy as np
import pytesseract
from PIL import Image
from docx import Document
import fitz
import time

router = APIRouter(prefix="/api/syllabus", tags=["Syllabus"])

# ---------- Настройка логирования ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Проверка Tesseract ----------
try:
<<<<<<< HEAD
    tesseract_path = os.getenv("TESSERACT_PATH")
<<<<<<< HEAD
    pytesseract.pytesseract.tesseract_cmd = tesseract_path # Update this path for the server
=======
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"  # Update this path for the server
>>>>>>> 54fff93 (generate)
=======
    pytesseract.pytesseract.tesseract_cmd = tesseract_path  # Update this path for the server
>>>>>>> dbb1288 (Объединил два эндпойнта в один)
    pytesseract.get_tesseract_version()
    logger.info("Tesseract is accessible")
except Exception as e:
    logger.error(f"Tesseract setup failed: {e}")
    raise RuntimeError(f"Tesseract setup failed: {e}")


# ---------- Pydantic модели ----------
class SyllabusRequest(BaseModel):
    token: str
    webservices_url: str = "https://moodle.astanait.edu.kz/webservice/rest/server.php"


# ---------- Moodle клиент ----------
class MoodleClient:
    def __init__(self, webservices_url: str, ws_token: str, user_id: Optional[int] = None):
        self.webservices_url = webservices_url
        self.ws_token = ws_token
        self.user_id = user_id

    def _make_request(self, ws_function: str, params: Dict = None) -> Dict:
        if params is None:
            params = {}
        params.update({
            'wstoken': self.ws_token,
            'wsfunction': ws_function,
            'moodlewsrestformat': 'json'
        })
        response = requests.post(self.webservices_url, data=params)
        response.raise_for_status()
        data = response.json()
        if 'exception' in data:
            raise Exception(data['message'])
        return data

    def get_user_courses(self) -> List[Dict]:
        return self._make_request('core_course_get_enrolled_courses_by_timeline_classification',
                                  {'classification': 'inprogress'})['courses']

    def get_course_contents(self, course_id: int) -> List[Dict]:
        return self._make_request('core_course_get_contents', {'courseid': course_id})

    def download_file(self, file_url: str, save_path: str) -> None:
        response = requests.get(file_url, params={'token': self.ws_token}, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


# ---------- Загрузчик силлабусов ----------
class SyllabusDownloader:
    def __init__(self, download_dir: str):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def download_syllabuses(self, client: MoodleClient) -> List[str]:
        syllabuses = []
        courses = client.get_user_courses()
        for course in courses:
            course_id = course['id']
            contents = client.get_course_contents(course_id)
            for section in contents:
                for module in section.get('modules', []):
                    if module.get('modname') == 'resource' or module.get('modname') == 'folder':
                        for file_info in module.get('contents', []):
                            file_name = file_info['filename']
                            if file_name.lower().endswith(
                                    ('.pdf', '.doc', '.docx')) and 'syllabus' in file_name.lower():
                                file_url = file_info['fileurl']
                                save_path = os.path.join(self.download_dir, file_name)
                                client.download_file(file_url, save_path)
                                syllabuses.append(save_path)
        return syllabuses


# ---------- Проверка документов на штампы и подписи ----------
DEBUG_FOLDER = "debug_stamps"
os.makedirs(DEBUG_FOLDER, exist_ok=True)


# Helper function to clean up debug images
def cleanup_debug_folder():
    """Removes all files from the debug folder to save disk space"""
    if os.path.exists(DEBUG_FOLDER):
        for file in os.listdir(DEBUG_FOLDER):
            file_path = os.path.join(DEBUG_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Deleted debug file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        logger.info(f"Debug folder cleaned up: {DEBUG_FOLDER}")


def extract_images_from_pdf(pdf_path):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    except Exception as e:
        logger.error(f"Error opening PDF {pdf_path}: {e}")
    return images


def extract_images_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    img = np.ones((2000, 1600), dtype=np.uint8) * 255
    y0 = 50
    for i, line in enumerate(full_text.split('\n')):
        y = y0 + i * 30
        if y >= img.shape[0] - 50:
            break
        cv2.putText(img, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return [img_bgr]


def filter_logo(contours, blue_mask, gray):
    filtered_contours = []
    logo_regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:  # Relaxed minimum area
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        if not (0.5 < aspect_ratio < 1.5):  # More relaxed aspect ratio for stamps
            continue
        x_extended = max(0, x - 20)
        y_extended = max(0, y - 20)
        w_extended = min(gray.shape[1] - x_extended, w + 40)
        h_extended = min(gray.shape[0] - y_extended, h + 40)
        roi = gray[y_extended:y_extended + h_extended, x_extended:x_extended + w_extended]
        if roi.size == 0:
            continue
        text = pytesseract.image_to_string(roi, lang='eng').lower()
        logo_keywords = ['astana', 'university', 'университет', 'logo', 'emblem', 'it', 'aitu']
        if any(kw in text for kw in logo_keywords):
            logo_regions.append((x_extended, y_extended, w_extended, h_extended))
        else:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            if radius > 25:  # Relaxed radius requirement
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                blue_pixels = cv2.bitwise_and(blue_mask, mask)
                blue_ratio = np.count_nonzero(blue_pixels) / np.count_nonzero(mask) if np.count_nonzero(mask) > 0 else 0
                if blue_ratio > 0.2:  # More relaxed blue threshold
                    filtered_contours.append(cnt)
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:  # Relaxed minimum area
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0
        if not (0.5 < aspect_ratio < 1.5):  # More relaxed aspect ratio
            continue
        is_overlapping = False
        for l_x, l_y, l_w, l_h in logo_regions:
            if (x < l_x + l_w and x + w > l_x and y < l_y + l_h and y + h > l_y):
                is_overlapping = True
                break
        if not is_overlapping:
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
            if (len(approx) > 5 and radius > 15) or radius > 30:  # More relaxed requirements
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], 0, 255, -1)
                blue_pixels = cv2.bitwise_and(blue_mask, mask)
                blue_ratio = np.count_nonzero(blue_pixels) / np.count_nonzero(mask) if np.count_nonzero(mask) > 0 else 0
                if blue_ratio > 0.15:  # More relaxed blue threshold
                    filtered_contours.append(cnt)
    return filtered_contours


def detect_signature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 40 < area < 1200:  # More relaxed area constraints
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
            if len(approx) > 3:  # More relaxed approximation
                signature_contours.append(cnt)
    return len(signature_contours) > 4  # Relaxed requirement for signature contours


def detect_concentric_patterns(mask, min_rings=3):  # Relaxed concentric pattern requirement
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    hist = cv2.calcHist([dist * 255], [0], None, [50], [1, 255])
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 3:  # Relaxed peak height
            peaks.append(i)
    return len(peaks) >= min_rings


def detect_stamp(image):
    debug_original = image.copy()
    h, w = image.shape[:2]
    right_boundary = int(w * 0.6)
    right_part = image[:, right_boundary:]
    debug_right_part = right_part.copy()
    hsv = cv2.cvtColor(right_part, cv2.COLOR_BGR2HSV)

    # Expanded blue range to catch more variations of blue
    lower_blue = np.array([90, 50, 50])  # More relaxed blue range
    upper_blue = np.array([130, 255, 255])  # Wider hue range
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Also detect blue-like colors that might appear in scanned documents
    lower_blue_gray = np.array([100, 20, 50])  # For faded blue in scans
    upper_blue_gray = np.array([140, 100, 200])
    blue_gray_mask = cv2.inRange(hsv, lower_blue_gray, upper_blue_gray)

    # Combine the masks
    blue_mask = cv2.bitwise_or(blue_mask, blue_gray_mask)

    # Add morphological operations to enhance blue regions
    kernel = np.ones((3, 3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    gray = cv2.cvtColor(right_part, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    combined_mask = cv2.bitwise_or(thresh, blue_mask)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_contours = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
    stamp_contours = filter_logo(contours, blue_mask, gray)
    for cnt in stamp_contours:
        cv2.drawContours(debug_contours, [cnt], 0, (0, 0, 255), 2)
    debug_blue = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    has_signature = detect_signature(right_part)
    text = pytesseract.image_to_string(right_part, lang='rus+eng').lower()

    # Expanded stamp keyword list to catch more variations
    stamp_keywords = ['печать', 'м.п.', 'seal', 'stamp', 'подпись', 'signature', 'approved', 'утверждено']
    text_has_stamp_indicators = any(kw in text for kw in stamp_keywords)
    approved_text = 'approved' in text or 'утверждено' in text or 'подтверждено' in text

    definite_stamp_contours = []
    for cnt in stamp_contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if area < 400:  # Slight relaxation
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        blue_pixels = cv2.bitwise_and(blue_mask, mask)
        blue_ratio = np.count_nonzero(blue_pixels) / np.count_nonzero(mask) if np.count_nonzero(mask) > 0 else 0
        has_concentric = detect_concentric_patterns(mask)

        # More relaxed criteria for definite stamps
        if (circularity > 0.7 and blue_ratio > 0.2 and radius > 25) or (has_concentric and blue_ratio > 0.15):
            definite_stamp_contours.append(cnt)

    stamp_like_objects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:  # Relaxed minimum area
            continue
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        has_concentric_pattern = detect_concentric_patterns(mask)
        blue_pixels = cv2.bitwise_and(blue_mask, mask)
        blue_ratio = np.count_nonzero(blue_pixels) / np.count_nonzero(mask) if np.count_nonzero(mask) > 0 else 0
        (_, _), radius = cv2.minEnclosingCircle(cnt)

        # More relaxed criteria for probable stamps
        if (circularity > 0.65 and blue_ratio > 0.15 and radius > 20) or (has_concentric_pattern and blue_ratio > 0.1):
            stamp_like_objects.append(cnt)

    has_definite_stamp = len(definite_stamp_contours) > 0
    has_probable_stamp = len(stamp_like_objects) > 0 and (text_has_stamp_indicators or has_signature)
    approval_section_exists = approved_text

    # Consider probable stamps as valid when there's supporting evidence
    has_stamp = has_definite_stamp or (has_probable_stamp and (has_signature or approval_section_exists))

    debug_info = {
        "original": debug_original,
        "right_part": debug_right_part,
        "threshold": thresh,
        "blue_mask": debug_blue,
        "contours": debug_contours
    }

    logger.info(
        f"Stamp detection: definite={has_definite_stamp}, probable={has_probable_stamp}, approval={approval_section_exists}, signature={has_signature}, final={has_stamp}")

    return has_stamp, has_signature, debug_info


def analyze_document(file_path):
    file_name = os.path.basename(file_path)
    logger.info(f"Processing file: {file_name}")
    try:
        if file_path.endswith('.pdf'):
            images = extract_images_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            images = extract_images_from_docx(file_path)
        else:
            return f"Unsupported format: {file_name}", [], []
        if not images or len(images) == 0:
            return f"No pages extracted: {file_name}", [], []

        first_page = images[0]
        h, w = first_page.shape[:2]

        # Check more of the page area to find stamps that might be located in different positions
        upper_part = first_page[0:int(h * 0.5), :]  # Extended to 50% of the page

        has_stamp, has_signature, debug_images = detect_stamp(upper_part)

        base_name = os.path.splitext(file_name)[0]
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_original.png"), debug_images["original"])
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_right_part.png"), debug_images["right_part"])
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_threshold.png"), debug_images["threshold"])
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_blue_mask.png"), debug_images["blue_mask"])
        cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_contours.png"), debug_images["contours"])

        # If we didn't find a stamp in the upper part, check the whole first page
        if not has_stamp:
            logger.info(f"No stamp found in upper part, checking full page for {file_name}")
            has_stamp, more_signatures, more_debug_images = detect_stamp(first_page)
            has_signature = has_signature or more_signatures

            # Save additional debug images
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_full_original.png"), more_debug_images["original"])
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_full_blue_mask.png"), more_debug_images["blue_mask"])
            cv2.imwrite(os.path.join(DEBUG_FOLDER, f"{base_name}_full_contours.png"), more_debug_images["contours"])

        issues = []
        if not has_stamp:
            issues.append("Stamp missing on first page")
        if not has_signature:
            issues.append("Signature missing on first page")

        return f"Processed: {file_name}", issues, [file_name] if not issues else []
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        return f"Error processing {file_name}: {e}", [f"Error: {e}"], []


def verify_documents(folder_path):
    logger.info(f"Verifying folder: {folder_path}")
    if not os.path.isdir(folder_path):
        logger.error("Invalid or non-existent folder path")
        return "Invalid or non-existent folder path"
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if f.lower().endswith(('.pdf', '.docx')) and not f.startswith('~$')]
    logger.info(f"Found files: {files}")
    if not files:
        logger.warning("No PDF or DOCX files found in folder")
        return "No PDF or DOCX files found in folder"
    detailed_missing_report = {}
    good_documents = []
    results = []
    for file_path in files:
        result, issues, valid_docs = analyze_document(file_path)
        if issues:
            detailed_missing_report[os.path.basename(file_path)] = issues
        if valid_docs:
            good_documents.extend(valid_docs)
        results.append(result)
    logger.info(f"Verification results: {results}")
    final_report = "## Verification Results\n\n"
    if detailed_missing_report:
        final_report += "### Documents with Issues:\n"
        for doc, issues in detailed_missing_report.items():
            final_report += f"- **{doc}**:\n" + "\n".join(f"  - {issue}" for issue in issues) + "\n"
    else:
        final_report += "All documents have stamp and signature.\n"
    if good_documents:
        final_report += "\n### Valid Documents:\n"
        for doc in good_documents:
            final_report += f"- **{doc}**\n"
    logger.info(f"Final report: {final_report}")

    # Clean up debug images after verification
    cleanup_debug_folder()

    return final_report


# ---------- FastAPI endpoint ----------
@router.post("/process-syllabuses")
async def process_syllabuses(request: SyllabusRequest):
    """
    Downloads syllabuses from Moodle using the provided token and verifies if they have stamps and signatures.
    """
    try:
        # Validate token
        if not request.token or len(request.token) < 32:
            raise HTTPException(status_code=400, detail="Invalid or missing Moodle token")

        # Create temporary directory for the syllabuses
        temp_dir = f"./temp_syllabuses/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Created temporary directory: {temp_dir}")

        try:
            # Initialize client and downloader
            client = MoodleClient(request.webservices_url, request.token)
            downloader = SyllabusDownloader(temp_dir)

            # Download syllabuses
            syllabuses = downloader.download_syllabuses(client)
            logger.info(f"Downloaded {len(syllabuses)} syllabuses: {syllabuses}")

            # Ensure files are written before verification
            time.sleep(1)  # Temporary delay to ensure files are fully written

            if not syllabuses:
                return {
                    "message": "No syllabuses found",
                    "files": [],
                    "verification_result": "No files to verify"
                }

            # Verify the documents
            verification_result = verify_documents(temp_dir)

            response = {
                "message": "Syllabuses downloaded and verified",
                "files": [os.path.basename(path) for path in syllabuses],
                "verification_result": verification_result
            }

            return response
        finally:
            # Clean up the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Removed temporary directory: {temp_dir}")

            # Make sure debug images are cleaned up
            cleanup_debug_folder()

    except Exception as e:
        # Clean up debug folder in case of error
        cleanup_debug_folder()

        logger.error(f"Error processing syllabuses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing syllabuses: {str(e)}")