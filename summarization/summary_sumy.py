from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# Models
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer

file = './test.txt'
parser = PlaintextParser.from_file(file, Tokenizer('english'))
doc = parser.document


def print_out(sentences):
    for sent in sentences:
        print(sent)
    print()


lex = LexRankSummarizer()
print_out(lex(doc, 2))

lsa = LsaSummarizer()
print_out(lsa(doc, 2))

luhn = LuhnSummarizer()
print_out(luhn(doc, 2))

# Best for the current task
kl = KLSummarizer()
print_out(kl(doc, 2))
