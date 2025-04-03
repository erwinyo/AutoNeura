from doctr.io import DocumentFile

def do_ocr_on_pdf(document, model):
    doc = DocumentFile.from_pdf(document)
    result = model(doc)
    
    return result


def do_ocr_on_image(document, model):
    doc = DocumentFile.from_images(document)
    result = model(doc)
    
    return result