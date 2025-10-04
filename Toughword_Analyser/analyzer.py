import fitz
from toughness import get_top_tough_words

def extract_paragraphs_from_pdf(filepath):
    doc = fitz.open(filepath)
    paragraphs = []

    for page in doc:

        text = page.get_text("text")


        lines = text.split('\n')
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue 

            buffer += " " + line


            if line.endswith('.'):
                paragraphs.append(buffer.strip())
                buffer = "" 


        if buffer:
            paragraphs.append(buffer.strip())

    return paragraphs



def analyze_pdf(filepath):
    paragraphs = extract_paragraphs_from_pdf(filepath)
    for i, para in enumerate(paragraphs):
        tough_words = get_top_tough_words(para)
        print(f"\nParagraph {i+1}:\n{para}\nTop 3 tough words: {tough_words}")

if __name__ == "__main__":
    analyze_pdf("uploads/TestFile.pdf")
