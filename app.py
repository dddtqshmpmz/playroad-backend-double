import base64
import json
import os
import shutil
import time
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sockets import Sockets

from engine import run

try:
    # Delete history files before restarting app
    shutil.rmtree('./scripts/')
    os.makedirs('./scripts')
except:
    pass

app = Flask(__name__)
CORS(app)
sockets = Sockets(app)

ws_clients = {}
ws_messages = {}


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/maps')
def get_maps():
    maps = []
    for filename in os.listdir('./maps'):
        with open('./maps/' + filename) as f:
            maps.append({'name': filename, 'content': json.load(f)})
    return jsonify(sorted(maps, key=lambda x: x['name']))


@app.route('/reset', methods=['POST'])
def reset():
    req = request.get_json()
    if ws_clients.get(request.remote_addr) is not None and not ws_clients.get(request.remote_addr).closed:
        msg = """{{
          "Event": "update_pos",
          "position": [{0},{1},{2}]
        }}""".format(*req["position"])
        ws = ws_clients[request.remote_addr]
        ws.send(msg)
    return "ok"


@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        filename = str(time.time()).replace('.', '-')
        req = request.get_json()
        with open('scripts/{0}.py'.format(filename), 'w') as f:
            f.write(req['code'])
        f = __import__('scripts.' + filename)
        script = getattr(f, filename)

        view = json.loads(ws_messages[request.remote_addr])["view1"]
        nparr = np.frombuffer(base64.b64decode(view), np.uint8)
        view = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        view1 = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        view = json.loads(ws_messages[request.remote_addr])["view2"]
        nparr = np.frombuffer(base64.b64decode(view), np.uint8)
        view2 = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

        res = run(script.image_to_speed, int(req['step']), req['position'], script.log, view1, view2)
        pos = res["position"]
        pos = [x for x in pos]
        if ws_clients.get(request.remote_addr) is not None and not ws_clients.get(request.remote_addr).closed:
            msg = """{{
              "Event": "update_pos",
              "position": [{0},{1},{2}]
            }}""".format(*pos)
            ws = ws_clients[request.remote_addr]
            ws.send(msg)
        return jsonify(res)
    except Exception:
        err = traceback.format_exc()
        err = err.replace("\n", "<br>")
        err = err.replace(" ", "&nbsp;")
        return jsonify({'Error': err})


@sockets.route('/ws')
def echo_socket(ws):
    while not ws.closed:
        ws_clients[request.remote_addr] = ws
        message = ws.receive()
        ws_messages[request.remote_addr] = message


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
