from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
lemma = WordNetLemmatizer()
stemmer = PorterStemmer()
stem = SnowballStemmer("english")
print(lemma.lemmatize("Identified"))
print(stemmer.stem("identified"))
print(stem.stem("identified"))
def get_counts(sequence):
    # 所有值被初始化为0
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts
