# -*- coding: utf-8 -*-
"""
@author: Niyazi Ulke
"""

from collections import Counter
import os
from math import log2 
from random import randint

def train_count(path):
    """ Extracts training data from corresponding files. """
    vocab = set() 
    legit_terms = Counter() # counter for number of each term in legit docs
    spam_terms = Counter() # counter for number of each term in spam docs
    legit_doc_freq = Counter() # counter for document frequencies among legit docs
    spam_doc_freq = Counter() # counter for document frequencies among spam docs
    legit_docs = 0 # number of legit documents
    spam_docs = 0 # number of spam documents
   
    path_legit = path + "/training/legitimate"
    
    for root, dirs, filenames in os.walk(path_legit):
        for filename in filenames:
           if filename.endswith('.txt'):
               filepath = path_legit + "/" + filename
               file = open(filepath,"r")
               text = file.read().split() # read and split the email
               legit_doc_freq.update(set(text)) # update document frequency of each word
               legit_terms.update(text) # update number of each term
               vocab = vocab | set(text) # update the vocabulary
               legit_docs += 1
               file.close()
                              
    path_spam = path + "/training/spam" 
    for root, dirs, filenames in os.walk(path_spam):
        for filename in filenames:
           if filename.endswith('.txt'):
               filepath = path_spam + "/" + filename
               file = open(filepath,"r")
               text = file.read().split() # read and split the email
               spam_doc_freq.update(set(text)) # update document frequency of each word
               spam_terms.update(text) # update number of each term
               vocab = vocab | set(text) # update the vocabulary
               spam_docs += 1
               file.close()
  
    return (legit_terms, spam_terms, legit_doc_freq, spam_doc_freq, legit_docs, spam_docs, vocab)

def test_count(path):
    """ Counts terms in each test file"""
    legit_docs = [] # stores term frequencies of each legit file
    spam_docs = [] # stores term frequencies of each spam file
    
    path_legit = path + "/test/legitimate"
    for root, dirs, filenames in os.walk(path_legit):
        for filename in filenames:
           if filename.endswith('.txt'):
               filepath = path_legit + "/" + filename
               file = open(filepath,"r")
               text = file.read().split() 
               legit_docs.append(Counter(text)) # append term frequency dictionary to the list
               file.close()
               
    path_spam = path + "/test/spam"
    for root, dirs, filenames in os.walk(path_spam):
        for filename in filenames:
           if filename.endswith('.txt'):
               filepath = path_spam + "/" + filename
               file = open(filepath,"r")
               text = file.read().split()
               spam_docs.append(Counter(text)) # append term frequency dictionary to the list
               file.close()    
    return (legit_docs, spam_docs)

def model1_train(params):
    """ Trains model 1, using all terms as features"""
    legit_terms, spam_terms, _ , _ , legit_docs, spam_docs, vocab = params # read parameters needed for model 1
    legit_count = sum(legit_terms.values()) # number of terms in legit files
    spam_count = sum(spam_terms.values()) # number of terms in spam files
    legit_prob = {}
    spam_prob = {}
    
    for term in vocab: # calculate the probabilities with Laplace smoothing, alpha = 1
        if term in legit_terms:
            legit_prob[term] = (legit_terms[term] + 1 ) / (legit_count + len(vocab))
        else:
            legit_prob[term] = (0 + 1) / (legit_count + len(vocab))
    
        if term in spam_terms:
            spam_prob[term] = (spam_terms[term] + 1) / (spam_count + len(vocab))
        else:
            spam_prob[term] = (0 + 1) / (spam_count + len(vocab))

    legit_coefficient = legit_docs / (spam_docs + legit_docs) # P(c_legit)
    spam_coefficient = spam_docs / (spam_docs + legit_docs) # P(c_spam)
    print("Size of vocabulary for model 1 is: " , len(vocab) , "\n")
    return (legit_prob, spam_prob, legit_coefficient, spam_coefficient, vocab) # return information of the model

def model2_train(params):
    """ Trains model 2, selecting k=100 terms as features"""
    legit_terms, spam_terms, legit_doc_freq, spam_doc_freq, legit_docs, spam_docs, vocab = params # read parameters needed for model 2
    total_docs = legit_docs + spam_docs
    term_score = []
    
    #
    # MI implementation
    #
    
    for word in vocab: 
        if word in legit_doc_freq: # legit_1 is the number of legit documents that contain the word.
            legit_1 = legit_doc_freq[word] 
        else:
            legit_1 = 0
        
        if word in spam_doc_freq: # spam_1 is the number of spam documents that contain the word.
            spam_1 = spam_doc_freq[word]
        else:
            spam_1 = 0
            
        legit_0 = legit_docs - legit_1
        spam_0 = spam_docs - spam_1
        all_0 = legit_0 + spam_0
        all_1 = legit_1 + spam_1
        score = 0
        
        # Application of MI formula. There are 2 classes, calculation for once is enough for a word.
        if legit_1 != 0:
            score += (legit_1 / total_docs)* log2((legit_1 / total_docs)/ (all_1 / total_docs * legit_docs / total_docs)) 
        if legit_0 != 0:
            score += (legit_0 / total_docs)* log2((legit_0 / total_docs)/ (all_0 / total_docs * legit_docs / total_docs))          
        if spam_1 != 0:
            score += (spam_1 / total_docs)* log2((spam_1 / total_docs)/ (all_1 / total_docs * spam_docs / total_docs)) 
        if spam_0 != 0:
            score += (spam_0 / total_docs)* log2((spam_0 / total_docs)/ (all_0 / total_docs * spam_docs / total_docs))
        
        
        term_score.append((word, score)) # tuples list in form : (word, MI score)
  
    
    terms_sorted = sorted(term_score, key = lambda x: -x[1]) 
    
    
    vocab_selected = [ pair[0] for pair in terms_sorted[:100]] # selected vocabulary
    
    legit_selected = dict((term, legit_terms[term]) for term in vocab_selected if term in legit_terms) # selected subset of legit terms counter
    spam_selected = dict((term, spam_terms[term]) for term in vocab_selected if term in spam_terms) # selected subset of spam terms counter
    legit_count = sum(legit_selected.values()) # number of selected terms in legit documents
    spam_count = sum(spam_selected.values()) # number of selected terms in spam documents
    
    legit_prob = {}
    spam_prob = {}
    
    for term in vocab_selected: # calculate the probabilities with Laplace smoothing, alpha = 1 , k = 100 features are selected. 
        if term in legit_terms:
            legit_prob[term] = (legit_selected[term] + 1 ) / (legit_count + len(vocab_selected))
        else:
            legit_prob[term] = (0 + 1) / (legit_count + len(vocab_selected))
    
        if term in spam_terms:
            spam_prob[term] = (spam_terms[term] + 1) / (spam_count + len(vocab_selected))
        else:
            spam_prob[term] = (0 + 1) / (spam_count + len(vocab_selected))         

    legit_coefficient = legit_docs / (spam_docs + legit_docs) # P(c_legit)
    spam_coefficient = spam_docs / (spam_docs + legit_docs) # P(c_spam)
    print("Selected terms for the second model are: ", vocab_selected , "\n")
    return (legit_prob, spam_prob, legit_coefficient, spam_coefficient, vocab_selected) # return information of the model
    
    
def predict(model, documents):
    """ Makes predictions on test data using the model"""
    legit_prob, spam_prob, legit_coefficient, spam_coefficient, vocab = model # extract parameters of model
    legit_docs, spam_docs = documents # seperate legit and spam documents

    legit_predictions = [] # list to store each prediction
    for doc in legit_docs:
        legit_log = 0 # log probability of being legit
        spam_log = 0 # log probability of being spam
        for term in doc:
            if term in vocab: # calculate probabilities only with terms from the vocab
                legit_log += log2(legit_prob[term]) * doc[term]
                spam_log  += log2(spam_prob[term]) * doc[term]           

        legit_log += log2(legit_coefficient) 
        spam_log += log2(spam_coefficient)
        if legit_log >= spam_log:         
            # when log probabilities are equal, also choose legit.
            # note : the same is done for actual spam documents, knowing the correct label does not affect how the model works. 
            # otherwise, this would be a fradulent model.
            legit_predictions.append("legit")
        else:
            legit_predictions.append("spam")
    spam_predictions = []
    for doc in spam_docs:
        legit_log = 0 # log probability of being legit
        spam_log = 0 # log probability of being spam
        for term in doc:
            if term in vocab: # calculate probabilities only with terms from the vocab
                legit_log += log2(legit_prob[term]) * doc[term]
                spam_log  += log2(spam_prob[term]) * doc[term]           
          
        legit_log += log2(legit_coefficient)
        spam_log += log2(spam_coefficient)
        
        if legit_log >= spam_log:
            spam_predictions.append("legit")
        else:
            spam_predictions.append("spam")
    return (legit_predictions , spam_predictions) # return prediction results


def evaluate(predictions):
    """ Calculates scores for the predictions"""
    legit_predictions , spam_predictions = predictions
    
    true_legit = legit_predictions.count("legit") # count true legit predictions
    false_legit = spam_predictions.count("legit") # count false legit predictions
    
    true_spam = spam_predictions.count("spam") # count true spam predictions
    false_spam = legit_predictions.count("spam") # count false spam predictions
    
    # calculate precision, recall and f scores for each class
    # calculate macro-averaged scores     
    legit_prec = true_legit / (true_legit + false_legit)
    spam_prec = true_spam / (true_spam + false_spam)
    macro_prec = (legit_prec + spam_prec) / 2
    
    legit_recall = true_legit / (true_legit + false_spam)
    spam_recall = true_spam/ (true_spam + false_legit)
    macro_recall = (legit_recall + spam_recall) / 2
    
    legit_f = 2* legit_prec* legit_recall / (legit_prec + legit_recall)
    spam_f = 2* spam_prec* spam_recall / (spam_prec + spam_recall)
    macro_f = (legit_f + spam_f) / 2

    
    return {'legit_prec' : legit_prec, 'spam_prec' : spam_prec, 'macro_prec' : macro_prec,
            'legit_recall' : legit_recall, 'spam_recall' : spam_prec, 'macro_recall' : macro_recall,
            'legit_f' : legit_f, 'spam_f' : spam_f, 'macro_f' : macro_f }
   
    

def random_test(predictions_A, predictions_B, sig = 0.05 , R = 1000):
    """ Randomization test between predictions_A and predictions_B"""
    f_A = evaluate(predictions_A)['macro_f'] 
    f_B = evaluate(predictions_B)['macro_f']
  
    s = abs(f_A - f_B)
    legit_pred_A , spam_pred_A = predictions_A # seperate predictions of legit and spam documents
    legit_pred_B , spam_pred_B = predictions_B
    count = 0
    for i in range(R): 
        
        legit_pred_A2 = [] # will store legit-labeled half of A' 
        spam_pred_A2 = [] # will store spam-labeled half of A'
        legit_pred_B2 = [] 
        spam_pred_B2 = []
        
        for j in range(len(legit_pred_A)):
            shuffle = randint(0,1) # with a probability of 0.5, shuffle two predictions
            if shuffle == 1:
                legit_pred_A2.append(legit_pred_B[j])
                legit_pred_B2.append(legit_pred_A[j])
                
            else: # do not shuffle
                legit_pred_A2.append(legit_pred_A[j])
                legit_pred_B2.append(legit_pred_B[j])
                
        for j in range(len(spam_pred_A)):
            shuffle = randint(0,1) # with a probability of 0.5, shuffle two predictions
            if shuffle == 1:
                spam_pred_A2.append(spam_pred_B[j])
                spam_pred_B2.append(spam_pred_A[j])
                
            else: # do not shuffle
                spam_pred_A2.append(spam_pred_A[j])
                spam_pred_B2.append(spam_pred_B[j])
                
        f_A2 = evaluate((legit_pred_A2, spam_pred_A2))['macro_f'] # get f score of A'
        f_B2 = evaluate((legit_pred_B2, spam_pred_B2))['macro_f'] # get f score of B'
        s_2 = abs(f_A2 - f_B2) # calculate s*
        if s_2 >= s:
            count += 1
    p = (count + 1) / (R + 1) # calculate p
    print("Randomization test result: p=", p)
    if p < sig : 
        print("The difference between two models is significant.")
    else : 
        print("The difference between two models is not significant.")
        
    
# =============================================================================
def main():
    path = os.curdir + "/dataset"
    train_data = train_count(path) # extract data from training documents
    model1 = model1_train(train_data) # train model1
    model2 = model2_train(train_data) # train model2
    documents = test_count(path) # extract data from test documents
    predictions1 = predict(model1,documents) # make predictions with model1
    predictions2 = predict(model2,documents) # make predictions with model2
    model1_evaluation = evaluate(predictions1) 
    print("Evaluation of model 1: " , model1_evaluation, "\n")
    model2_evaluation = evaluate(predictions2) 
    print("Evaluation of model 2: " , model2_evaluation, "\n")
    random_test(predictions1, predictions2) # randomization test to see if two models predict significantly different

if __name__ == "__main__":
    main()
