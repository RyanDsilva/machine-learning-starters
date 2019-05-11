from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest
from string import punctuation

SUMMARY_SENT = 3

with open('./test.txt', 'r') as file:
  content = file.read()

tokens = word_tokenize(content.lower())
sentences = sent_tokenize(content)
stop_words = set(stopwords.words('english') + list(punctuation))
words = [word for word in tokens if (word.isalpha() and word not in stop_words)]

word_freq = FreqDist(words)
rankings = defaultdict(int)

for i, sentence in enumerate(sentences):
  for word in word_tokenize(sentence.lower()):
    if word in word_freq:
      rankings[i] += word_freq[word]

indexes = nlargest(SUMMARY_SENT, rankings, key=rankings.get)
final_sentences = [sentences[j] for j in sorted(indexes)]
summary = ' '.join(final_sentences) 

print(summary)