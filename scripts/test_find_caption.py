from core.circuit_store import find_caption_page_in_pdf

# sample caption text from paper 2407.04826 (Figure 1)
caption = "MCT gates with positive condition (closed circle) and negative condition (open circle)"
res = find_caption_page_in_pdf('2407.04826', caption)
print('Detected:', res)
