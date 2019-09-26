from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('digimonhome.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    sent = request.form
    favorite = sent["search"].capitalize()
    datadf = pd.read_json('digimon.json')
    datadf2 = datadf[["digimon", "stage", "type", "attribute"]]

    def kombinasi(i):
        return str(i['stage']) + '$' + str(i['type']) + '$' + str(i['attribute'])
    datadf2['comb'] = datadf2.apply(kombinasi, axis=1)

    model = CountVectorizer(
        tokenizer = lambda data: data.split('$')
    )
    sim = model.fit_transform(datadf2['comb'])

    if len(datadf2[datadf2["digimon"] == favorite].index.values) == 0:
        return render_template('digierror.html') 

    index = datadf2[datadf2["digimon"] == favorite].index.values[0]
    score = cosine_similarity(sim)
# nama gambar stage type attribute
    digLst = list(enumerate(score[index]))
    digSort = sorted(digLst, key=lambda a: a[1], reverse=True)
    r = [] #recommended
    for d in digSort[:7]:
        temp=[]
        if datadf.iloc[d[0]]['digimon'] != favorite:
            name = datadf.iloc[d[0]]['digimon']
            gbr = datadf.iloc[d[0]]['image']
            stg = datadf.iloc[d[0]]['stage']
            typ = datadf.iloc[d[0]]['type']
            atr = datadf.iloc[d[0]]['attribute']
            temp.extend([name, gbr, stg, typ, atr])
            r.append(temp)

    favgbr = datadf[datadf['digimon']==favorite]['image'].values[0]
    favstg = datadf[datadf['digimon']==favorite]['stage'].values[0]
    favtyp = datadf[datadf['digimon']==favorite]['type'].values[0]
    favatr = datadf[datadf['digimon']==favorite]['attribute'].values[0]
    z = []
    z.extend([favorite, favgbr, favstg, favtyp, favatr])
    r.append(z)
    return render_template('digiresult.html',
    name1=r[0][0], gbr1=r[0][1], stg1=r[0][2], typ1=r[0][3], atr1=r[0][4],
    name2=r[1][0], gbr2=r[1][1], stg2=r[1][2], typ2=r[1][3], atr2=r[1][4],
    name3=r[2][0], gbr3=r[2][1], stg3=r[2][2], typ3=r[2][3], atr3=r[2][4],
    name4=r[3][0], gbr4=r[3][1], stg4=r[3][2], typ4=r[3][3], atr4=r[3][4],
    name5=r[4][0], gbr5=r[4][1], stg5=r[4][2], typ5=r[4][3], atr5=r[4][4],
    name6=r[5][0], gbr6=r[5][1], stg6=r[5][2], typ6=r[5][3], atr6=r[5][4],
    namefav=r[6][0], gbrfav=r[6][1], stgfav=r[6][2], typfav=r[6][3], atrfav=r[6][4],
    )
    
if __name__ == '__main__':
    app.run(debug=True)