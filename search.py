# -*- coding: utf-8 -*-
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "clean.txt"
TOP_K = 5

def load_chunks(path):
    text = Path(path).read_text(encoding="utf-8")
    raw = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 100]
    return raw

def main():
    if not Path(DATA_FILE).exists():
        print("clean.txt 파일이 없습니다.")
        return

    chunks = load_chunks(DATA_FILE)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    X = vectorizer.fit_transform(chunks)

    print("문서 검색기 (exit 입력 시 종료)")
    while True:
        q = input("질문 > ")
        if q.lower() == "exit":
            break
        qv = vectorizer.transform([q])
        sims = cosine_similarity(qv, X).ravel()
        top = sims.argsort()[::-1][:TOP_K]

        for i, idx in enumerate(top, 1):
            print(f"#{i} score={sims[idx]:.3f}")
            print(chunks[idx][:200].replace("\n"," "), "\n")

if __name__ == "__main__":
    main()
