from textblob import TextBlob

str = input("Enter a statement for analysis:\n")

analysis = TextBlob(str)
print(analysis.sentiment)
