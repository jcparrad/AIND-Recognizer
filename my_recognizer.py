import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    all_sequences = test_set.get_all_sequences()
    for index, sequence in all_sequences.items():
        X, lengths= test_set.get_item_Xlengths(index)
        probabilities_guess_dict = {}

        best_guess_word = None
        best_score = float("-inf")

        for word, model_word in models.items():
            try:
                score_word= model_word.score(X, lengths)
                probabilities_guess_dict[word]= score_word
                if score_word > best_score:
                    best_score = score_word
                    best_guess_word = word

            except:
                probabilities_guess_dict[word]= float('-inf')

        guesses.append(best_guess_word)
        probabilities.append(probabilities_guess_dict)

    return probabilities, guesses
