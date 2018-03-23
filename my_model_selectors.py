import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def bic_score(self, number_components):
        # Bayesian information criteria: BIC = −2 log L + p log N
        # L is the likelihood of the fitted mode
        # p is the number of parameters
        # N is the number of data points.
        model_bic = self.base_model(number_components)
        N = len(self.sequences)
        p = (number_components**2 + 2*number_components*model_bic.n_features - 1)
        logL = model_bic.score(self.X, self.lengths)
        BIC = -2*logL + p*math.log(N)
        return BIC, model_bic

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_bic = float("inf")
        bic_model = None
        for number_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                bic, model_bic = self.bic_score(number_components)
                if bic < best_bic:
                    best_bic = bic
                    bic_model = model_bic
            except:
                pass
        return bic_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def dic_score(self, number_components):
        # DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))
        model_dic = self.base_model(number_components)
        logl = model_dic.score(self.X, self.lengths)
        other_words_likelihood = []
        for this_word, (X, lengths) in self.hwords.items():
            if this_word != self.this_word:
                other_words_likelihood.append(model_dic.score(X, lengths))
        dic = logl -  np.mean(other_words_likelihood)
        return dic, model_dic

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_dic = float("-inf")
        dic_model = None
        for number_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                dic, model_dic = self.dic_score(number_components)
                if dic > best_dic:
                    best_dic = dic
                    dic_model = model_dic
            except:
                pass
        return dic_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''


    def cv_score(self, number_components):
        model_cv = self.base_model(number_components)
        split_method = KFold()
        scores_fold = []

        for _, test_idx in split_method.split(self.sequences):
            test_X, test_length = combine_sequences(test_idx, self.sequences)
            scores_fold.append(model_cv.score(test_X, test_length))

        cv = np.mean(scores_fold)
        return cv, model_cv

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_cv = float("inf")
            cv_model = None
            for number_components in range(self.min_n_components, self.max_n_components + 1):

                    cv, model_cv = self.cv_score(number_components)
                    if cv < best_cv:
                        best_cv = cv
                        cv_model = model_cv

            return cv_model
        except:
            return self.base_model(self.n_constant)
