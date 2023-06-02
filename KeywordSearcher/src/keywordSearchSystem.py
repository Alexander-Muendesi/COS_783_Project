import gensim
from gensim import corpora, models
import spacy
from src.fileReader import FileReader
from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')
class KeywordSearch:
    def __init__(self):
        self.fReader = FileReader()
        self.processedData = self.fReader.processEnron()
        self.tokenData = self.tokenizeData()
        # self.word2vecModel = gensim.models.Word2Vec(self.processedData,min_count=1)
        self.word2vecModel = gensim.models.Word2Vec(self.tokenData,min_count=1)
        # self.dictionary = corpora.Dictionary(self.processedData)#create a dictionary from the processed documents
        self.dictionary = corpora.Dictionary(self.tokenData)#create a dictionary from the processed documents
        # self.corpus = [self.dictionary.doc2bow(doc.split()) for doc in self.processedData]#convert the documents into a document term matrix
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.tokenData]#convert the documents into a document term matrix
        self.nlp = spacy.load('en_core_web_sm')
        self.entityList = self.extractEntities(self.processedData)

    def tokenizeData(self):
        tokens = []
        for data in self.processedData:
            tokens.append(word_tokenize(data))
            
        return tokens

    def extractEntities(self, documents):
        entities = []

        for doc in documents:
            doc_entities = []

            spacy_doc = self.nlp(doc)

            for entity in spacy_doc.ents:
                doc_entities.append(entity.text)
            
            entities.append(doc_entities)
        
        return entities

    def enhancedKeywordSearch(self,query,oneWord):
        #search using word embeddings
        similarWords = ""
        if oneWord:
            queryVector = self.word2vecModel.wv[query]
            similarWords = self.word2vecModel.wv.similar_by_vector(queryVector)

        #search using topic modelling
        queryBow = self.dictionary.doc2bow(query.lower().split())
        lda_model = models.LdaModel(self.corpus, num_topics=5, id2word=self.dictionary, passes=10)
        topicDistribution = lda_model.get_document_topics(queryBow)
        top_topics = sorted(topicDistribution, key=lambda x: x[1], reverse=True)[:3]
    

        #search using entity linking
        entityMatches = []
        for entities in self.entityList:
            entityMatches.extend([entity for entity in entities if all(word.lower() in entity.lower() for word in query.split())])
    
        #combine the results from each step
        enhanced_results = []
        enhanced_results.extend(similarWords)
        # enhanced_results.extend(top_topics)
        enhanced_results.extend(entityMatches)

        return enhanced_results