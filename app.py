from flask import Flask,render_template, request, send_from_directory
import math as m
import cv2 as cv
import numpy as np 
import mediapipe as mp
import statistics as s
import scipy.stats as stats
from scipy.fft import fft
from scipy.signal import find_peaks

app = Flask(__name__)

@app.route('/')
def index():
    data = {"name":"ekkarach",
            "age": 30,
            "alary": 50000}
    return render_template('index.html',mydata = data)

@app.route('/admin')
def profile():
    username = "P_each"
    name = "Ekkarach"
    age = "22"
    return render_template('admin.html',username = username,myname = name, myage = age)

@app.route('/user')#/user/<name>
def member():
    user = {"name","ID","PhoneNum"}
    return render_template('user.html',myuser = user)

@app.route('/insertVideo')
def insertV():
    return render_template('insertVideo.html')

@app.route('/calfft')
def cal_fft(video_path):
    
    return render_template('admin.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video part in the form'

    video = request.files['video']

    if video.filename == '':
        return 'No selected video file'

    # Save the video to the 'static' folder
    video_path ='static/' + video.filename
    video.save(video_path)
    iris_position = detect_iris(video_path) #item=f'Iris Position: {iris_position}'
    return render_template('success.html',  item=iris_position)

def detect_iris(video_path):

    RIGHT_IRIS=[474,475,476,477]
    LEFT_IRIS=[469,470,471,472]
    i=0
    count=1
    FF_freq = []
    freqfunc = [0]

    ampleftx = []
    amplefty = []
    amprightx = []
    amprighty = []

    left_anglex = []
    left_angley = []
    right_anglex = []
    right_angley = []

    n = 512 #sample size 512 256 128
    delta_t = 1/30 #sampling time
    delta_tXn = delta_t*n # this is used to find a freq
    camera = cv.VideoCapture(video_path)
    mp_face_mesh =  mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh( 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5 
    ) as face_mesh: 
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv.flip(frame, 1) 
            frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
            rgb_frame = cv.cvtColor(frame, cv. COLOR_BGR2RGB) 
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points=np.array(
                    [
                        np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                        for p in results.multi_face_landmarks[0].landmark
                    ]
                )
            
                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                left_anglex.append(l_cx)
                left_angley.append(l_cy)
                right_anglex.append(r_cx)
                right_angley.append(r_cy)
            
    camera.release()
    cv.destroyAllWindows()

    fft_leftx = fft(left_anglex)
    fft_lefty = fft(left_angley)
    fft_rightx = fft(right_anglex)
    fft_righty = fft(right_angley)
    while i < n:
        hp1x = abs(fft_leftx[i])
        hp1y = abs(fft_lefty[i])

        hp2x = abs(fft_rightx[i])
        hp2y = abs(fft_righty[i])

        ampleftx.append(hp1x)
        amplefty.append(hp1y)

        amprightx.append(hp2x)
        amprighty.append(hp2y)
        i+=1
    while count < n:
        hp0 = freqfunc[count-1]+(1/delta_tXn)
        freqfunc.append(float(format(hp0,".6f")))
        # freqfunc = np.append(freqfunc, hp0)
        count+=1
    freqfunc = freqfunc[:-12]

    ampleftx = ampleftx[12:]
    amplefty = amplefty[12:]
    amprightx = amprightx[12:]
    amprighty = amprighty[12:]
    
    for p in freqfunc:
        if p <= 5:
            FF_freq.append(p)
        else:
            break
    ampleftx= ampleftx[:len(FF_freq)]
    amplefty= amplefty[:len(FF_freq)]
    amprightx= amprightx[:len(FF_freq)]
    amprighty= amprighty[:len(FF_freq)]

    mean_peakleftx = s.mean(ampleftx)
    mean_peaklefty = s.mean(amplefty)
    mean_peakrightx = s.mean(amprightx)
    mean_peakrighty = s.mean(amprighty)

    std_peakleftx = s.stdev(ampleftx)
    std_peaklefty = s.stdev(amplefty)
    std_peakrightx = s.stdev(amprightx)
    std_peakrighty = s.stdev(amprighty)

    sqrtSampleSize = (m.sqrt(delta_tXn))
    temp1x = std_peakleftx/sqrtSampleSize
    temp1y = std_peaklefty/sqrtSampleSize
    temp2x = std_peakrightx/sqrtSampleSize
    temp2y = std_peakrighty/sqrtSampleSize

    CI_lx = stats.norm.interval(0.975, loc = mean_peakleftx, scale = temp1x)
    lwb_lx ,upb_lx = CI_lx
    CI_ly = stats.norm.interval(0.975, loc = mean_peaklefty, scale = temp1y)
    lwb_ly ,upb_ly = CI_ly

    CI_rx = stats.norm.interval(0.975, loc = mean_peakrightx, scale = temp2x)
    lwb_rx ,upb_rx = CI_rx
    CI_ry = stats.norm.interval(0.975, loc = mean_peakrighty, scale = temp2y)
    lwb_ry ,upb_ry = CI_ry

    return upb_lx,upb_ly,upb_rx,upb_ry,lwb_lx,lwb_ly,lwb_rx,lwb_ry

@app.route('/video/<filename>')
def display_video(filename):
    return render_template('display_video.html', filename=filename)

@app.route('/static/<filename>')
def uploaded_file(filename):
    return send_from_directory('static', filename)
#"yooo uer: {}, age : {}" .format(name,age)
if __name__ == "__main__":
    app.run(debug=True)