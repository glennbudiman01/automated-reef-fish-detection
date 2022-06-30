from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import requests
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/img/predict_img/'
# define labels
labels = ['Kakap', 'Pogot', 'Rengginan', 'Swanggi']

model = load_model('model_ikan.h5')


def preprocess(img_path, input_size):
    nimg = img_path.convert('RGB').resize(input_size, resample=0)
    img_arr = (np.array(nimg))/255
    return img_arr


def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


@app.route("/", methods=['GET', 'POST'])
def get_output():
    if request.method == 'GET':
        return render_template("index.html")

    elif request.method == 'POST':
        model = load_model('model_ikan.h5', compile=False)
        img = request.files['photo']
        img_path = app.config['UPLOAD_FOLDER']+img.filename
        img.save(img_path)
        im = Image.open(img_path)

        # read image
        input_size = (150, 150)
        X = preprocess(im, input_size)
        X = reshape([X])
        y = model.predict(X)

        if labels[np.argmax(y)] == 'Kakap':
            gambar = './static/img/gambar/kakap1.jpg'
            penjelasan = 'Kakap adalah keluarga ikan laut dasaran (demersal) yang hidup secara berkelompok di dasar-dasar karang atau terumbu karang. Kakap termasuk famili Lutjanidae. Mempunyai ciri tubuh yang bulat pipih dengan sirip memanjang sepanjang punggung. Jenis ikan kakap yang banyak ditemui di Indonesia adalah jenis kakap merah (L. campechanus). Beberapa jenis yang lain yang juga banyak ditemui adalah kakap kuning, kakap hitam dan lain-lain.'

        elif labels[np.argmax(y)] == 'Pogot':
            gambar = './static/img/gambar/pogot1.jpg'
            penjelasan = 'Ikan Pogot atau Triggerfish yang bernama ilmiah Aluterus hidupnya di perairan laut dalam yang berkarang. Bentuknya pipih seperti ikan bawal serta memiliki gigi yang besar dan tajam serta berkulit tebal sangat. Walaupun ciri fisiknya menyeramkan ternyata ikan ini cukup enak untuk dikonsumsi, menurut orang-orang yang pernah memakan ikan ini rasanya seperti ayam, sebab itu di sebagian daerah Indonesia menyebutnya ikan ayam-ayam. Ikan ini tergolong kedalam jenis ikan omnivora yaitu ikan pemakan segala.'

        elif labels[np.argmax(y)] == 'Rengginan':
            gambar = './static/img/gambar/swanggi1.jpeg'
            penjelasan = ''

        else:
            gambar = './static/img/gambar/swanggi1.jpeg'
            penjelasan = 'Ikan swanggi (Priacanthus tayenus Richardson, 1846) termasuk salah satu jenis dari ikan demersal yang umumnya mencari makan secara nokturnal dan diurnal dengan sama baiknya (Starnes 1984). Ikan ini juga bernilai ekonomis dan ekologis tetapi masih belum dikaji.'

        return render_template("predict.html", result=labels[np.argmax(y)], gambar=gambar, penjelasan=penjelasan)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
