import xml.etree.ElementTree as ET
import re
import os

INPUT_FILE = "./kowiki-latest-pages-articles.xml"
OUTPUT_DIR = "./data/korean_corpus/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    """Remove wiki markup and return plain text."""
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'={2,}[^=]+={2,}', '', text)
    text = re.sub(r'^\s*[\*#:;]+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    text = re.sub(r'[\[\]\{\}]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_sentences(text):
    """Split text into sentences."""
    text = clean_text(text)
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    korean_re = re.compile(r'[\uac00-\ud7af]')
    good = []
    for s in sentences:
        s = s.strip()
        if len(s) > 15 and len(s) < 500 and korean_re.search(s):
            good.append(s)
    return good

print(f"Reading {INPUT_FILE}...")
print("This may take several minutes for a full Wikipedia dump.")

file_count = 0
sentence_count = 0
outfile = None
sentences_per_file = 100000

context = ET.iterparse(INPUT_FILE, events=('end',))

for event, elem in context:
    if elem.tag.endswith('}text') or elem.tag == 'text':
        if elem.text:
            sentences = extract_sentences(elem.text)
            for sent in sentences:
                if outfile is None or sentence_count % sentences_per_file == 0:
                    if outfile:
                        outfile.close()
                    fname = os.path.join(OUTPUT_DIR, f"wiki_{file_count:04d}.txt")
                    outfile = open(fname, 'w', encoding='utf-8')
                    file_count += 1

                outfile.write(sent + '\n')
                sentence_count += 1

                if sentence_count % 100000 == 0:
                    print(f"  Extracted {sentence_count} sentences...")

        elem.clear()

if outfile:
    outfile.close()

print(f"\nDone! Extracted {sentence_count} sentences into {file_count} files in {OUTPUT_DIR}")