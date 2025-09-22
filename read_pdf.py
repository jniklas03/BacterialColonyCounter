from pypdf import PdfReader
import re

pattern_count = re.compile(r"Count :\s*(\d+)")
pattern_name = re.compile(r"Sample N°\s*:\s*.*?\bWt\s+(\d+)\s+Platte\s+(\d+)", re.DOTALL)

doc= PdfReader("tag0.pdf")

for page_idx, page in enumerate(doc.pages, start=0):
    text = page.extract_text()

    counts = pattern_count.findall(text)

    names = pattern_name.findall(text)

    for image_idx, image_file_object in enumerate(page.images[1:], start=1):

        filename = f"WT{names[0][0]}_P{names[0][1]}_counts{counts[0]}_img{image_idx}.png"
        with open(filename, "wb") as fp:
            fp.write(image_file_object.data)
