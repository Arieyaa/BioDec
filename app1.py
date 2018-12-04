import time
import random
from flask import Flask, render_template, url_for, request
from flask_bootstrap import Bootstrap 
#NLP
from textblob import TextBlob,Word 
import random
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords  
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

app = Flask(__name__)
bootstrap=Bootstrap(app)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
	if request.method=='POST':
		rawtext=request.form['rawtext']
		#NLP stuff
		
		########
		df = pd.read_csv('Data/dataCompiled.csv')
		dfRem = df[df["Class"]=='Remembering'].sample(n=100).reset_index(drop=True)
		dfUnd = df[df['Class']=='Understanding'].sample(n=100).reset_index(drop=True)#gets shuffled sample of each subset
		dfApp = df[df["Class"]=='Applying'].sample(n=100).reset_index(drop=True)
		dfAna = df[df["Class"]=='Analyzing'].sample(n=100).reset_index(drop=True)
		dfEva = df[df["Class"]=='Evaluating'].sample(n=100).reset_index(drop=True)
		dfCre = df[df["Class"]=='Creating'].sample(n=100).reset_index(drop=True)
		#df = pd.concat([dfRem, dfUnd,dfApp,dfAna,dfEva,dfCre])
		

		stopword_list = stopwords.words("english")


		stemmer = SnowballStemmer("english")
		sw_stem = []
		for i in stopword_list:
			if stemmer.stem(i) not in sw_stem:
				sw_stem.append(stemmer.stem(i))



		vec= TfidfVectorizer(stop_words=sw_stem)
		

		X = df['Question']
		X_vec = vec.fit_transform(X)
		y = df.Class


		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=234)

		clf = LinearSVC(random_state=0, tol=1e-5)
		d = {'col1': [rawtext]}
		X_test = pd.DataFrame(data=d)

		X_train_vect = vec.fit_transform(X_train)
		X_test_vect = vec.transform(X_test)
		clf.fit(X_train_vect, y_train)
		received_text2= clf.predict(X_test_vect)


	return render_template('index.html', received_text=received_text2, question=rawtext)

if __name__=='__main__':
	app.run(debug=True)
