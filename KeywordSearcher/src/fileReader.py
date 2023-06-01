import pandas as pd
from src.textPreprocessor import TextPreprocessor

class FileReader:
    def __init__(self):
        self.textProcessor = TextPreprocessor()

    def processEnron(self):
        preprocessed_emails = []

        #load the enron email data set into a Pandas Data frame
        enron_emails = pd.read_csv('emails.csv') 

        #preprocess each email in the data set
        count = 0
        for email in enron_emails['message']:
            if count > 100:
                break
            count += 1
            #extract the message content from the email
            message_start = email.find("X-FileName:") + len("X-FileName:")

            message = email[message_start:].strip()
            preprocessed_text = self.textProcessor.preprocess_text(message)
            preprocessed_emails.append(preprocessed_text)
            # print(preprocessed_text)
        return preprocessed_emails
    
    def processHateSpeech(self):
        preprocessed_speech = []

        #load the hate speech data set
        hateSpeech = pd.read_csv("labeled_data.csv")

        for item in hateSpeech.itertuples(index=False):
            preprocessed_text = self.textProcessor.preprocess_text(item[6])
            preprocessed_speech.append(preprocessed_text)

        return preprocessed_speech