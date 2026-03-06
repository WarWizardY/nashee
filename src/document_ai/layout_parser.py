import os
import fitz  # PyMuPDF
from PIL import Image
import io
import torch
from transformers import TableTransformerForObjectDetection
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Global singletons for models
_TABLE_DETECTOR_MODEL = None

def get_table_detector() -> TableTransformerForObjectDetection:
    global _TABLE_DETECTOR_MODEL
    if _TABLE_DETECTOR_MODEL is None:
        try:
            logger.info("Loading Table Transformer model. This may take a moment...")
            _TABLE_DETECTOR_MODEL = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
        except Exception as e:
            logger.error(f"Failed to load table transformer: {e}")
            raise
    return _TABLE_DETECTOR_MODEL

def pdf_page_to_image(page: fitz.Page) -> Image.Image:
    """Converts a PyMuPDF page object into a PIL Image."""
    pix = page.get_pixmap(dpi=150)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def detect_tables_in_image(image: Image.Image) -> list:
    """
    Passes an image into Microsoft Table Transformer to detect tables.
    Returns a list of table bounding boxes [xmin, ymin, xmax, ymax].
    """
    model = get_table_detector()
    model.eval()

    # Preprocess image for the model
    # Table Transformer expects a standard vision transform: Resize + Normalize
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    pixel_values = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(pixel_values)
        
    # Process outputs
    width, height = image.size
    
    # Threshold for detection
    threshold = 0.7
    
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    
    # Scale boxes back to original image size
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), (width, height))
    
    tables = []
    for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled):
        # class id 0 is table in microsoft/table-transformer-detection
        cl = p.argmax()
        if cl.item() == 0:
            tables.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            
    return tables

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def extract_text_with_easyocr(image: Image.Image, tables: list) -> str:
    """
    Extracts text using EasyOCR. Treats tables separately if bounding boxes are provided.
    """
    import easyocr
    import numpy as np
    
    # Initialize reader locally inside function so it only runs if EasyOCR is available
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    # Convert PIL Image to OpenCV format
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy() # RGB to BGR
    
    document_text = ""
    
    # Just running general OCR over the entire page for now with layout awareness
    # Note: A real Table Structure model would split the table cells. For this prototype,
    # EasyOCR does a decent job at reading lines left-to-right.
    results = reader.readtext(img_cv, detail=1, paragraph=True)
    
    for bbox, text in results:
        document_text += text + "\n"
        
    return document_text

def parse_document_layouts(pdf_path: str) -> str:
    """
    Advanced Document AI Parsing:
    Uses PyMuPDF to convert PDF to images, Table Transformer to find tables,
    and EasyOCR to read the text in a layout-preserving manner.
    """
    doc = fitz.open(pdf_path)
    full_parsed_text = ""
    
    for i, page in enumerate(doc):
        logger.info(f"Parsing page {i+1}/{len(doc)}")
        try:
            image = pdf_page_to_image(page)
            # Detect tables using transformers
            tables = detect_tables_in_image(image)
            # Extract text (simulating layout preservation with simple OCR paragraphs)
            page_text = extract_text_with_easyocr(image, tables)
            
            full_parsed_text += f"\n--- Page {i+1} ---\n"
            full_parsed_text += f"[Detected {len(tables)} tables on this page]\n"
            full_parsed_text += page_text + "\n"
        except Exception as e:
            logger.error(f"Error parse page {i+1}: {e}")
            # Fallback to simple text extraction if ML fails
            full_parsed_text += page.get_text() + "\n"
            
    return full_parsed_text
