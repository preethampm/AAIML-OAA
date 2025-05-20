import easyocr
from transformers import pipeline

# Initialize OCR
reader = easyocr.Reader(['en'])

# Load Simplification (Paraphrasing) model
simplifier = pipeline("text2text-generation", model="t5-base")

def extract_text_from_image(image_path):
    result = reader.readtext(image_path, detail=0)
    return ' '.join(result)

def simplify_text(text):
    result = simplifier(f"simplify: {text}", max_length=100, do_sample=False)
    return result[0]['generated_text']

def main():
    image_path = input("Enter path to medical image: ")

    print("\nExtracting text from image...")
    extracted_text = extract_text_from_image(image_path)
    

    
    summarized_text = simplify_text(extracted_text)
    print(f"Summarised Text:\n{summarized_text}")

if __name__ == "__main__":
    main()
