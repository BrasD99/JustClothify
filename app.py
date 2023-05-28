from flask import Flask, render_template, request, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import os
import uuid
from helpers.processor import TextureProcessor
from helpers.methods import download_model


app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
data_folder = 'data'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['DATA_FOLDER'] = data_folder

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('hello'))

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        process_guid = str(uuid.uuid4())

        person_image = request.files['person_image']
        person_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{process_guid}-person.jpg")
        person_image.save(person_image_path)

        clothes_image = request.files['clothes_image']
        clothes_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{process_guid}-clothes.jpg")
        clothes_image.save(clothes_image_path)

        return redirect(url_for('process_images', process_guid=process_guid))

    return render_template('hello.html')

@app.route('/process/<process_guid>')
def process_images(process_guid):
    person_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{process_guid}-person.jpg")
    clothes_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{process_guid}-clothes.jpg")

    request = {
            'output_path': app.config['UPLOAD_FOLDER'],
            'temp_path': app.config['UPLOAD_FOLDER'],
            'densepose':{
                'config': os.path.join(app.config['DATA_FOLDER'], 'config.yaml'),
                'weights': os.path.join(app.config['DATA_FOLDER'], 'weights.pkl')
            },
            'models': [clothes_image_path],
            'person': person_image_path
    }

    processor = TextureProcessor(request)
    processor.extract(process_guid)

    return redirect(url_for('show_results', process_guid=process_guid))

@app.route('/results/<process_guid>')
def show_results(process_guid):
    person_image_path = url_for('static', filename=f"uploads/{process_guid}-person.jpg")
    clothes_image_path = url_for('static', filename=f"uploads/{process_guid}-clothes.jpg")
    result_image_path = url_for('static', filename=f"uploads/{process_guid}-result.jpg")
    return render_template('results.html', 
        person_image_path=person_image_path, 
        clothes_image_path=clothes_image_path, 
        result_image_path=result_image_path)

if __name__ == '__main__':
    if not os.path.exists(app.config['DATA_FOLDER']):
        download_model(app.config['DATA_FOLDER'])
    app.run(debug=True)