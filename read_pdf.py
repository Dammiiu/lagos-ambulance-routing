import fitz
import sys
import os

doc = fitz.open('THOMAS_Chapter_1-3.pdf')
print(f'Total pages: {len(doc)}')
all_text = []
for i, page in enumerate(doc):
    text = page.get_text()
    all_text.append(f'\n{"="*60}\nPAGE {i+1}\n{"="*60}\n{text}')
doc.close()

full = '\n'.join(all_text)
# Write to file to avoid encoding issues in terminal
with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    f.write(full)
print(f'Written {len(full)} chars to pdf_content.txt')
