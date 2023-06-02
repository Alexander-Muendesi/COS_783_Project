# BEFORE USING run 
        # pip install nltk in terminal
# Then run 
        # import nltk
        # nltk.download('wordnet')

# Then comment it out and use 



from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym not in synonyms:
                synonyms.append(synonym)
    return synonyms


# if __name__ == "__main__":
#     word = input("Enter a word: ")
#     synonyms = get_synonyms(word)
#     print("Synonyms:", synonyms)

