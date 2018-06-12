import face_recognition
import cv2
import numpy as np
import glob
import urllib.request
import math
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# functions definitions
def load_known_faces():
    path_list = glob.glob('known_people/*.jpg')
    known_face_names = []
    known_face_encodings = []
    not_loaded_faces = []
    # Create list of face names and face encodings
    for img_path in path_list:
        tmp_img = face_recognition.load_image_file(img_path)           # Load picture of known person
        try:
            tmp_face_encoding = face_recognition.face_encodings(tmp_img)[0]   # Get face encoding 
            known_face_names.append(img_path[13:-4])       # Add file name as face name ***13 is char count in known_people/ and -4 is .jpg***
            known_face_encodings.append(tmp_face_encoding)      # Add face encoding
        except:
            not_loaded_faces.append(img_path[13:-4])
    print("Loaded Faces ", known_face_names)
    print("Not Loaded Faces ", not_loaded_faces)
    return known_face_encodings, known_face_names

def init_vid(src):
    global source, url
    source = src
    if (source == None) or (source == "webcam"):
        global video_capture
        video_capture = cv2.VideoCapture(0)
    elif source == "ip_cam":
        #url='http://10.184.61.234:8080/shot.jpg'
        url='http://10.194.60.99:8080/shot.jpg'
        # url='http://10.194.30.23:8080/shot.jpg'

def get_vid_frame():
    global video_capture, source, url
    # print(source)
    if (source == None) or (source == "webcam"):
        ret, frame = video_capture.read()
    elif source == "ip_cam":
        imgResp=urllib.request.urlopen(url)
        imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
        frame=cv2.imdecode(imgNp,-1)
    return frame

def stop_vid():
    global video_capture
    if (source == None) or (source == "webcam"):
        video_capture.release()

def load_new_user(vid_frame):
    global user_index, known_face_encodings, known_face_names
    addr = "known_people/user_"+str(user_index)+".jpg"
    frame = vid_frame
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    user_img = rgb_small_frame
    print(user_index)
    try:
        user_face_encoding = face_recognition.face_encodings(user_img)[0]
    except IndexError:
        print("Face not found")
        return known_face_encodings, known_face_names

    cv2.imwrite(addr, vid_frame)
    known_face_encodings.append(user_face_encoding)
    known_face_names.append("user_"+str(user_index))

    user_index+=1
    print("Face Added")
    return known_face_encodings, known_face_names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
user_index = 0

# Initialize/Load modules
known_face_encodings, known_face_names = load_known_faces()
init_vid("webcam") # Initiate Video Feed

while True:
    # Grab a single frame of video
    orig_frame = get_vid_frame()
    frame = np.copy(orig_frame)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_index, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

        # Facial Features
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)

        for face_landmarks in face_landmarks_list:
            # Print the location of each facial feature in this image
            facial_features = [
                'top_lip',
                'bottom_lip'
            ]

            for facial_feature in facial_features:
                pt_list = []
                for pt_index, pt in enumerate(face_landmarks[facial_feature]):
                    l_pt = [p*4 for p in list(pt)]
                    # print("==========")
                    # print(l_pt)
                    pt_list.append(l_pt)
                # print(face_list)
                # print(face_list)
                pts = np.array(pt_list, np.int32)
                pts = pts.reshape((-1,1,2))
                print(pts)
                if facial_feature == 'top_lip':
                    top_lip_pts = pt_list
                    print('Top')
                elif facial_feature == 'bottom_lip':
                    bot_lip_pts = pt_list
                    print('Bottom')
                # print(pts)
                if facial_feature == 'top_lip':
                    for pt_num, circle in enumerate(pts):
                        crc = tuple([tuple(c) for c in circle])[0]
                        # crc = crc2[8:10]
                        # print(crc2)
                        # print('Upper Lip')
                        # print(type(crc))
                        # print(crc.shape)
                        # print(crc)
                        # if pt_num > -1 and pt_num < 13:
                        # 	cv2.circle(frame, crc, 4, (0,255,0))
                        # 	font = cv2.FONT_HERSHEY_DUPLEX
                        # 	cv2.putText(frame, str(pt_num), crc, font, 1.0, (255, 255, 255), 1)
                        if pt_num > -1 and pt_num < 13:
                            cv2.circle(frame, crc, 4, (255,255,0))
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, str(pt_num), crc, font, 0.5, (255, 255, 255), 1)
                    cv2.polylines(frame, [pts], False, (255,0,0), 2)
                else:
                    for pt_num, circle in enumerate(pts):
                        crc = [tuple(c) for c in circle][0]
                        # print(crc)
                        # cv2.circle(frame, crc, 4, (0,255,0))
                        if pt_num > -1 and pt_num < 13:
                        	cv2.circle(frame, crc, 4, (0,255,0))
                        	font = cv2.FONT_HERSHEY_DUPLEX
                        	cv2.putText(frame, str(pt_num), crc, font, 0.5, (255, 255, 255), 1)
                    cv2.polylines(frame, [pts], False, (0,0,255), 2)
            l_c = math.sqrt((top_lip_pts[0][0] - bot_lip_pts[6][0])**2 + (top_lip_pts[0][1] - bot_lip_pts[6][1])**2)
            r_c = math.sqrt((top_lip_pts[6][0] - bot_lip_pts[0][0])**2 + (top_lip_pts[6][1] - bot_lip_pts[0][1])**2)
            l_m = math.sqrt((top_lip_pts[10][0] - bot_lip_pts[8][0])**2 + (top_lip_pts[10][1] - bot_lip_pts[8][1])**2)
            c_m = math.sqrt((top_lip_pts[9][0] - bot_lip_pts[9][0])**2 + (top_lip_pts[9][1] - bot_lip_pts[9][1])**2)
            r_m = math.sqrt((top_lip_pts[8][0] - bot_lip_pts[10][0])**2 + (top_lip_pts[8][1] - bot_lip_pts[10][1])**2)
            l_r = math.sqrt((top_lip_pts[0][0] - bot_lip_pts[0][0])**2 + (top_lip_pts[0][1] - bot_lip_pts[0][1])**2)
            print("Left corner = {}".format(l_c))
            print("Right corner = {}".format(r_c))
            print("Left mid = {}".format(l_m))
            print("Center mid = {}".format(c_m))
            print("Right mid = {}".format(r_m))
            print("L to R = {}".format(l_r))
            ratio = (l_m + c_m + r_m ) / l_r
            print('Mouth Ratio is {}'.format(ratio))

            open_thresh = 0.2
            close_thresh = 0.1
            mouth_stat = ''     # Reset each time
            if ratio > open_thresh:
                mouth_stat = 'Open'
                prev_mouth_stat = 'Open'
            elif ratio < close_thresh:
                mouth_stat = 'Close'
                prev_mouth_stat = 'Close'
            elif ratio < open_thresh and ratio > close_thresh:
                mouth_stat = prev_mouth_stat
            print("Mouth is {}".format(mouth_stat))
        # Facial Features End

    process_this_frame = True


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        known_face_encodings, known_face_names = load_new_user(orig_frame)

# Release handle to the webcam
stop_vid()
cv2.destroyAllWindows()


