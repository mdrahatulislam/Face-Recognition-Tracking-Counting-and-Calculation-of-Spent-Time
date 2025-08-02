import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import face_recognition

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
#detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# Only enable it if you are using OpenVino environment



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def main():
    cap = cv2.VideoCapture("C:\\Users\\admin\\Desktop\\file1\\testface1_rotated.mp4")
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []
    # Load a sample picture and learn how to recognize it.

    # Load a second sample picture and learn how to recognize it.
    supri_image = face_recognition.load_image_file("Supri.jpg")
    supri_face_encoding = face_recognition.face_encodings(supri_image)[0]

    # Load a second sample picture and learn how to recognize it.
    rahat_image = face_recognition.load_image_file("Islam Md Rahatul.jpg")
    rahat_face_encoding = face_recognition.face_encodings(rahat_image)[0]

    # Load a second sample picture and learn how to recognize it.
    horio_image = face_recognition.load_image_file("Horio.jpg")
    horio_face_encoding = face_recognition.face_encodings(horio_image)[0]

    # Load a second sample picture and learn how to recognize it.
    hiroki_image = face_recognition.load_image_file("Hiroki.jpg")
    hiroki_face_encoding = face_recognition.face_encodings(hiroki_image)[0]

    # Load a second sample picture and learn how to recognize it.
    yamada_image = face_recognition.load_image_file("Yamada.jpg")
    yamada_face_encoding = face_recognition.face_encodings(yamada_image)[0]

    # Load a second sample picture and learn how to recognize it.
    nakatami_image = face_recognition.load_image_file("Nakatami.jpg")
    nakatami_face_encoding = face_recognition.face_encodings(nakatami_image)[0]


    # Create arrays of known face encodings and their names
    known_face_encodings = [
        supri_face_encoding,
        rahat_face_encoding,
        horio_face_encoding,
        hiroki_face_encoding,
        yamada_face_encoding,
        nakatami_face_encoding
    ]
    known_face_names = [
        "Supri",
        "Md Rahatul Islam",
        "Horio",
        "Hiroki",
        "Yamada",
        "Nakatami"
    ]

    while True:
        ret, frame = cap.read()
        if ret:
            # describe the type of
            # font you want to display
            font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

            # Get date and time and
            # save it inside a variable
            dt = str(datetime.datetime.now())
            frame = cv2.putText(frame, dt,
                                (10, 100),
                                font, 2,
                                (0, 0, 0),
                                4, cv2.LINE_8)

        frame = imutils.resize(frame, width=1400)
        total_frames = total_frames + 1
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time



        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        lpc_txt = "Live Person Count: {}".format(lpc_count)
        opc_txt = "Total Person Count: {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 1)
        cv2.putText(frame, opc_txt, (5, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
