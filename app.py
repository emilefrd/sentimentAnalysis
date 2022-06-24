import os
# prevent useless warnings from tf lib
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from flask import Flask, request, render_template
from gensim.utils import simple_preprocess
import transformers
from transformers import pipeline 
from utils import write_file, get_models
import fasttext
# remove useless warnings from transformers lib
transformers.logging.set_verbosity_error()


app = Flask(__name__)
tokenizer, model = get_models()


@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    
    # get text data
    text1 = request.form['text1']

    # store it to txt file
    write_file(text1)

    ########################Fasttext inference########################
    # cleand_text =  simple_preprocess(text1)
    # cleand_text = ' '.join([word for word in cleand_text])
    
    # model = fasttext.load_model("models/fasttext_model_quant.ftz")
    # pred = model.predict(text1)

    #######################Camembert inference########################    
    # create pipeline and perform inference
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    pred = nlp(text1)

    return render_template('form.html', text1=text1, label=pred[0]['label'], score=round(pred[0]['score'], 4))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002, threaded=True)
