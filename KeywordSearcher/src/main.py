from src.keywordSearchSystem import KeywordSearch
from src.semanticAnalyzer import analyze_sentiment
from src.synonymFinder import get_synonyms

# import nltk
# nltk.download('vader_lexicon')

searcher = KeywordSearch()
# result = searcher.enhancedKeywordSearch("reschedule")
# result = searcher.enhancedKeywordSearch("payment")

while True:
    print("Option 1: Find synonyms.")
    print("Option 2: Search for keywords.")
    print("Option 3: exit")
    option = input("Select the numerical value of an option: ")

    if option == "1":
        word = input("Enter a word for which you want synonyms: ")
        synonyms = get_synonyms(word)
        for s in synonyms:
            print(s)
    elif option == "2":
        word = input("Enter a keyword: ")
        result = ""
        if len(word) == 1:
            result = searcher.enhancedKeywordSearch(word,True)
        else:
            result = searcher.enhancedKeywordSearch(word,False)

        for r in result:
            result = "(Sentiment: "
            result += analyze_sentiment(r) + ")\t" + r
            print(result)
    elif option == "3":
        break
    print("\n")

#keywords for demo
#1. hunter 2. password 3. mark