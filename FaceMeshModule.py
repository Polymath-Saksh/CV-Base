import cv2
import mediapipe as mp
import time

class FaceMesh:
    def __init__(self, staticMode = False, maxFaces=2, detectionCon=0.5, trackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=self.maxFaces,
         min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawSpec, connection_drawing_spec=self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN,
                         0.5, (0, 255, 0), 1)
                        face.append([id, cx, cy])

                    faces.append(face)
        return img


def main():
    #cap = cv2.VideoCapture('Face-Mesh\\FaceVideos\\3.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    faceMesh = FaceMesh()

    while True:
        success, img = cap.read()

        img = faceMesh.findFaceMesh(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{str(int(fps))}', 
        (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.imshow("Face", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()