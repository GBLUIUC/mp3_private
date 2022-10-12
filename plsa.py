import numpy as np
import math

#from torch import norm


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.number_of_topics = 0 # I added this myself

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        
        num_docs = 0
        # Data filepath is stored in self.documents_path 
        f = open(self.documents_path) 
        
        # Each line is a document with space separated words
        for doc in f.readlines():
            terms = doc.split(' ')

            # Remove the 0\t and \n from the beginning and end 
            terms[0] = terms[0][2:]
            terms = terms[:-1]

            self.documents.append(terms)
            num_docs += 1

        f.close() 

        self.number_of_documents = num_docs 

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        
        # For every word in every document, add it to the vocabulary if we haven't seen it before 
        for doc in self.documents:
            for word in doc:
                if word not in self.vocabulary:
                    self.vocabulary.append(word)

        self.vocabulary_size = len(self.vocabulary)


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        
        # Initialize to zeros 
        self.term_doc_matrix = np.zeros([self.number_of_documents, self.vocabulary_size])

        for i in range(self.number_of_documents):
            doc = self.documents[i]
            for j in range(self.vocabulary_size):
                vocab_word = self.vocabulary[j]
                count = 0

                for doc_word in doc:
                    if doc_word == vocab_word:
                        count += 1

                self.term_doc_matrix[i][j] = count


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################

        self.number_of_topics = number_of_topics

        # Randomize using np.random.random_sample 
        self.document_topic_prob = np.random.random_sample((self.number_of_documents, self.number_of_topics))
        self.topic_word_prob = np.random.random_sample((self.number_of_topics, self.vocabulary_size))

        # Don't forget to normalize (using the normalize function provided with starter code)
        self.document_topic_prob = normalize(self.document_topic_prob)
        self.topic_word_prob = normalize(self.topic_word_prob)
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        # ############################

        # Expecation step: Update the self.topic_prob P(z | w, d)
        # Intuitively, you basically just multiply document-topic probability with word-topic probability

        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                topic_sum = 0 # Sum up the column for normalization
                for z in range(self.number_of_topics):
                    self.topic_prob[d][z][w] = self.document_topic_prob[d][z] * self.topic_word_prob[z][w]    # Reminder: Topic prob ordering is d z w
                    topic_sum += self.topic_prob[d][z][w]

                for z in range(self.number_of_topics):
                    self.topic_prob[d][z][w] /= topic_sum 


        ## Normalize across topic rows 
        # for d in range(self.number_of_documents):
        #     for z in range(self.number_of_topics):
        #         for w in range(self.vocabulary_size):
        #             self.topic_prob[d][z][w] /= topic_sums[z]
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        self.number_of_topics = number_of_topics
        # update P(w | z) aka self.topic_word_prob
        
        # ############################
        # your code here
        # ############################

        for w in range(self.vocabulary_size):
            for z in range(self.number_of_topics):
                prob = 0

                for d in range(self.number_of_documents):
                    self.topic_word_prob[z][w] += self.term_doc_matrix[d][w] * self.topic_prob[d][z][w] # Reminder to self: topic_prob is in d z w order

        # Don't forget to normalize
        self.topic_word_prob = normalize(self.topic_word_prob)

        
        # update P(z | d) aka self.document_topic_prob

        # ############################
        # your code here
        # ############################

        for d in range(self.number_of_documents):
            for z in range(self.number_of_topics):
                self.document_topic_prob[d][z] = 0

                for w in range(self.vocabulary_size):
                    self.document_topic_prob[d][z] += self.term_doc_matrix[d][w] * self.topic_prob[d][z][w]# Reminder to self: topic_prob is in d z w order

        # Don't forget to normalize
        self.document_topic_prob = normalize(self.document_topic_prob)

        

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        self.number_of_topics = number_of_topics
        # ############################
        # your code here
        # ############################

        # Sum up all the log likelihoods to get the overall likelihood 

        likelihood = 0
        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                topic_likelihood = 0
                for z in range(self.number_of_topics):
                    topic_likelihood += self.document_topic_prob[d][z] * self.topic_word_prob[z][w]   # Topic likelihood is P(d|z) * P(w|z)

                # Then we want to add up the log likelihoods weighted by the term doc counts
                if topic_likelihood > 0:
                    likelihood += np.log(topic_likelihood) * self.term_doc_matrix[d][w]

        self.likelihoods.append(likelihood)

        return

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        self.number_of_topics = number_of_topics

        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################

            # Expectation, maximization, likelihood, and then repeat
            self.expectation_step()
            self.maximization_step(self.number_of_topics)
            self.calculate_likelihood(self.number_of_topics)

            # If the likelihood gain is smaller than epsilon, PLSA is done
            if iteration > 1 and self.likelihoods[iteration] - self.likelihoods[iteration - 1] < epsilon:
                return




def main():
    documents_path = 'data/test.txt'
    #documents_path = 'data/DBLP.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
