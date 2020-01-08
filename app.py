import json
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sockets import Sockets

from engine import run

# Delete history files before restarting app
shutil.rmtree('./scripts/')
shutil.rmtree('./static/imgs/')
os.makedirs('./scripts')
os.makedirs('./static/imgs')

app = Flask(__name__)
CORS(app)
sockets = Sockets(app)

ws_clients = {}


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


@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        filename = str(time.time()).replace('.', '-')
        req = request.get_json()
        with open('scripts/{0}.py'.format(filename), 'w') as f:
            f.write(req['code'])
        f = __import__('scripts.' + filename)
        script = getattr(f, filename)
        pos = req['position']
        pos = [x for x in pos]
        if ws_clients.get(request.remote_addr) is not None and not ws_clients.get(request.remote_addr).closed:
            msg = """{{
              "Event": "update_pos",
              "position": [{0},{1},{2}]
            }}""".format(*pos)
            ws = ws_clients[request.remote_addr]
            ws.send(msg)
        return jsonify(run(script.image_to_speed, int(req['step']), req['position'], req['map'], script.log))
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


if __name__ == '__main__':
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler

    server = pywsgi.WSGIServer(('', 5000), app, handler_class=WebSocketHandler)
    server.serve_forever()
