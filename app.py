#!/usr/bin/env python
from importlib import import_module
import os
import flask
from flask import Flask, render_template, Response

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)

# @app.route('/route-link')
# def changeRoute():
#     print('sdfasdf')
#     return render_template('index1.html')

@app.route('/')
def index1():
    return render_template('index.html');


# @app.route('/videocam1')
# def index():
#     """Video streaming home page."""
#     return render_template('index1.html')


@app.route("/videos/", methods=['POST'])
def goToVideos():
    print('called route to videos')
    return render_template('index1.html');

@app.route("/videos1/", methods=['POST'])
def goToVideos():
    print('called route to videos')
    return render_template('index1.html');

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def gen1(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')            


# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(gen(Camera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videocam1')
def video_picamera():
    """Video streaming route. Put this in the src attribute of an img tag."""
    #from mycodo.mycodo_flask.camera.camera_picamera import Camera
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/videocam2')
def video_picamera1():
    """Video streaming route. Put this in the src attribute of an img tag."""
    #from mycodo.mycodo_flask.camera.camera_picamera import Camera
    return Response(gen1(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=True)
