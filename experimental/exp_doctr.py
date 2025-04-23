from doctr.io import DocumentFile
from doctr.models import ocr_predictor

def main():
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images("../resources/images/document/document1.jpg")
    
    result = model(doc)
    print(result)
 
if __name__ == "__main__":  
    main()