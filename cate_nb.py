### DEPRECATED ### 


# import pandas as pd
# import numpy as np

# class NaiveBayesClassifier:
#     def __init__(self):
#         self.spam_prob = 0.5
#         self.notSpam_prob = 0.5
        
#     def loadData(self):
#         df = pd.read_csv('data/spam_or_not_spam.csv')
#         return df

#     def compute_initial_probs(self):
#         data = self.loadData()
#         nrecords = len(data)
        
#         counts = data['label'].value_counts()
#         spam_counts = counts[1]
#         not_spam_counts = counts[0]
        
#         total = spam_counts + not_spam_counts
        
#         self.spam_prob = spam_counts / total
#         self.notSpam_prob = not_spam_counts / total
        
#         print(self.spam_prob, self.notSpam_prob)
#         return self.spam_prob, self.notSpam_prob
        
# if __name__ == "__main__":
#     nbc = NaiveBayesClassifier()
    
#     df = pd.read_csv('data/spam_or_not_spam.csv')
#     print(df['email'][0])
    