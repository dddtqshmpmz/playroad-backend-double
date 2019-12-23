import json
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

from engine import run

# Delete history files before restarting app
shutil.rmtree('./scripts/')
shutil.rmtree('./static/imgs/')
os.makedirs('./scripts')
os.makedirs('./static/imgs')

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/maps')
def get_maps():
    maps = []
    for filename in os.listdir('./maps'):
        with open('./maps/' + filename) as f:
            maps.append({'name': filename, 'content': json.load(f)})
    return jsonify(maps)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        filename = str(time.time()).replace('.', '-')
        req = request.get_json()
        with open('scripts/{0}.py'.format(filename), 'w') as f:
            f.write(req['code'])
        f = __import__('scripts.' + filename)
        script = getattr(f, filename)
        return jsonify(run(script.image_to_speed, int(req['step']), req['position'], req['map'], script.log))
    except Exception:
        err = traceback.format_exc()
        err = err.replace("\n", "<br>")
        err = err.replace(" ", "&nbsp;")
        return jsonify({'Error': err})


if __name__ == '__main__':
    app.run()
