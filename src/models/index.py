import os
#from flask_ngrok import run_with_ngrok
from flask import Flask, request, send_file,make_response
from werkzeug.utils import secure_filename
import cv2
from loss_stage import driver
import base64
app = Flask(__name__, static_url_path='/static')
#run_with_ngrok(app)
if not os.path.exists(os.path.join(app.instance_path, 'uploads')):
    os.makedirs(os.path.join(app.instance_path, 'uploads'))


@app.route("/", methods=['POST'])
def startProcess():
    name = ""
    try:
        isthisFile = request.files.get('file')
        print(isthisFile)
        print(isthisFile.filename)

        #src = request.files['source']
        #param = request.args.get("todo")

        path = os.path.join(app.instance_path,
                            'uploads/', secure_filename(isthisFile.filename))
        isthisFile.save(path)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, gray)


    except:
        return "Source is empty", 400

    try:
        driver(path)
    except Exception as e:
        print(e)
        return "Some error occurred while processing", 400

    try:
        # os.remove(os.path.join(app.instance_path,
        #                        'uploads/', secure_filename(path)))

        # path = '4.png'
        # response = send_file(os.path.join(path), attachment_filename=path)
        # #response = send_file(os.path.join(path), attachment_filename=isthisFile.filename)
        # response.headers.add('Access-Control-Allow-Origin', '*')

        with open(os.path.join(path), "rb") as f:
            image_binary = f.read()

            response = make_response(base64.b64encode(image_binary))
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.set('Content-Type', 'image/gif')
            response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
            return response
        
        return response

    except:
        return "Some error occurred while returning image", 418


if __name__ == "__main__":
    app.run(threaded=False)