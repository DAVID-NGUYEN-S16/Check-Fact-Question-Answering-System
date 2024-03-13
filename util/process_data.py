import re
from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text: str, lower = False) -> str:    
    text = re.sub(r"['\",\.\?:\!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    if lower == True: text = text.lower()
    return text
def split_sentence(paragraph):
    context_list=[]
    if paragraph[-2:] == '\n\n':
        paragraph = paragraph[:-2]
    c = True
    start = 0
    while c:
        context = ""
        for i in range(start ,len(paragraph[:-2])):
            if paragraph[i] == ".":

              # Kiểm tra trường hợp "\n\n"
              if paragraph[i+1] == "\n":
                if paragraph[i+2].isalpha() and paragraph[i+2].isupper():
                    break
                context = context + paragraph[i]
                start = i + 1
                break

              # Kiểm tra trường hợp gặp " "
              if paragraph[i+1] == " ":

                context = context + paragraph[i]
                start = i + 1
                break
            context = context + paragraph[i]
            if i == len(paragraph[:-3]):
                start = i
        if start == len(paragraph[:-3]):
            context += paragraph[start+1:]
            c = False
        context = preprocess_text(context)
        if len(context.split())>2:
            context_list.append(context)
    return context_list
def process_data(text):
    
    
    line_text = split_sentence(text)


    return '. '.join(line_text)

