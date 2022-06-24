from datetime import datetime
from lib2to3.pgen2 import token
import numpy as np
from gensim.models import KeyedVectors
from transformers import CamembertTokenizer, TFCamembertForSequenceClassification


# get w2v model for vectorization
model = KeyedVectors.load_word2vec_format("./models/frWiki_no_phrase_no_postag_500_cbow_cut10.bin", binary=True)


def write_file(data):
    with open('./data/user_inputs.csv', 'a') as f:
        f.write(str(datetime.today()) + "; ")
        f.write(data + '\n')


def vectorize(tokenized_sentence):
    
    result = []
    for token in tokenized_sentence:
        if token in model: 
            result.append(model[token]) # get numpy vector of a word
    result = np.asarray(result)
    
    return result.mean(axis=0) # average of word vectors in sentence


def get_models():
    # tokenizer takes time too
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # loading models takes time
    model = TFCamembertForSequenceClassification.from_pretrained("jplu/tf-camembert-base")
    model.load_weights('models/camembert_weights.hdf5')
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return tokenizer, model