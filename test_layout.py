import fitz
from src.document_ai.layout_parser import parse_document_layouts

doc = fitz.open()
page = doc.new_page()
text = "IN THE HIGH COURT OF DELHI\nCompany: FakeTech Industries\nNotice of Default."
page.insert_text((50, 50), text, fontsize=12)

# Draw a mock "table" rectangle
page.draw_rect(fitz.Rect(50, 100, 400, 200), color=(0,0,0))
page.insert_text((60, 120), "Year | Revenue | Profit", fontsize=10)
page.insert_text((60, 140), "2023 | 50M     | 5M", fontsize=10)
page.insert_text((60, 160), "2024 | 40M     | -2M", fontsize=10)

doc.save("mock_doc.pdf")
print("Saved mock_doc.pdf")

print("Running Layout Parser...")
try:
    result = parse_document_layouts("mock_doc.pdf")
    print("Extraction Result:")
    print(result)
except Exception as e:
    print("Error during parsing:", e)
