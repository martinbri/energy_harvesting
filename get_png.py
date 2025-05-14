import sys
import os
import fitz  # PyMuPDF


def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extrait toutes les images intégrées d'un PDF et les enregistre en PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    img_count = 0

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)

        if not image_list:
            continue

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            output_filepath = os.path.join(
                output_dir,
                f"page{page_index+1}_img{img_index}.{image_ext}"
            )
            with open(output_filepath, "wb") as img_file:
                img_file.write(image_bytes)
            img_count += 1

    doc.close()
    print(f"{img_count} images extraites vers '{output_dir}'")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_images.py <pdf_path> <output_dir>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]

    extract_images_from_pdf(pdf_path, output_dir)
