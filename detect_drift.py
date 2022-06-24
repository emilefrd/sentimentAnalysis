import numpy as np
import pandas as pd
from scipy import stats
from gensim.utils import simple_preprocess
from utils import vectorize
import umap
from tqdm import tqdm
tqdm.pandas()


# load historical dataset
df = pd.read_pickle('./data/allocine_dataset.pickle')
df_train = df['train_set'].sample(n=100, random_state=1)
df_train['review']=df_train['review'].progress_apply(lambda x: simple_preprocess(x))
train_embeddings = df_train['review'].progress_apply(vectorize)
train_embeddings = np.array(train_embeddings.values.tolist())  

# load user inputs dataset
df_inputs = pd.read_csv('./data/user_inputs.csv',  sep=";", header=None)
df_inputs.columns = ["datetime", "review"]
df_inputs['review']=df_inputs['review'].progress_apply(lambda x: simple_preprocess(x))
input_embeddings = df_inputs['review'].progress_apply(vectorize)
input_embeddings = np.array(input_embeddings.values.tolist()) 

# then we need to perform dimensionality reduction in order to be able to compare components distribution 
trans = umap.UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=1).fit(train_embeddings)
umap_emb_train = trans.transform(train_embeddings)
umap_emb_inputs = trans.transform(input_embeddings)

# affecting 2d array to a dataframe
umap_emb_train =  pd.DataFrame(umap_emb_train)
umap_emb_inputs =  pd.DataFrame(umap_emb_inputs)

# K-S test is a nonparametric test that compares the cumulative distributions of two data sets
# null hp : same distribution, if the null is rejected -> distributions are != -> there is a covariate drift
nb_rejected = 0 
for col in umap_emb_train.columns:
    test = stats.ks_2samp(umap_emb_train[col], umap_emb_inputs[col])
    if test[1] < .05:
        nb_rejected += 1
        print(f"Column {col} got rejected, we might encounter a covariate shift! Please have a closer look to the user inputs!")

print(f"{nb_rejected} rejected columns among {len(umap_emb_train.columns)} in total!")

# NB: for NLP tasks, we might be better off using cosine similarity between the vectors from the train set and those from the input set
# if cosine similarity is too low, it would be a sign of covariate drift