from pypdf import PdfReader
import os
import re

def read_pdf(
        source,
        file,
        save_path,
        save_images=False
):
    save_path_extracted_pdf = save_path + r"\FromPDF"
    os.makedirs(save_path_extracted_pdf, exist_ok=True)

    pattern_count = re.compile(r"Count :\s*(\d+)")
    pattern_name = re.compile(r"(WT\s*\d+\s+P\s*\d+)")

    doc = PdfReader(source)

    all_counts = []
    all_names = []

    for page in doc.pages:
        text = page.extract_text()

        count = int(pattern_count.findall(text)[0])
        all_counts.append(count)

        name = str(pattern_name.findall(text)[0]).replace(" ", "_")
        all_names.append(name)

        if save_images:
            for image_idx, image_file_object in enumerate(page.images[1:2], start=1): #change the slice, and naming to get all images
                filename = f"{file}_{name}"
                with open(f"{os.path.join(save_path_extracted_pdf, filename)}.jpg", "wb") as fp:
                    fp.write(image_file_object.data)


    output = dict(zip(all_names, all_counts))

    with open(f"{os.path.join(save_path_extracted_pdf, file)}.txt", "w") as f:
        f.write(str(output))

    return(output)

read_pdf(
    source=r"PDFs\23.9.2025.pdf",
    file="23.9.2025.pdf",
    save_path=r"C:\Users\jakub\Documents\Bachelorarbeit\Code\160925",
    save_images=True
)