import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
PROTOTXT = "face_detector/deploy.prototxt"
CAFFEMODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5
class Facenet(nn.Module):
    def __init__(self):
        super(Facenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1,bias=False)#8,112,112
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0,bias=False)#16,110,110
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.AdaptiveMaxPool2d((55,55))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1,bias=False)#32,55,55
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0,bias=False)#64,53,53
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.AdaptiveMaxPool2d((27,27))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1,groups=64, padding=1,bias=False)#128,27,27
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1,groups=128, padding=0,bias=False)#256,25,25
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.AdaptiveMaxPool2d((13,13))
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1,groups=256, padding=1,bias=False)#512,13,13
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, stride=1,groups=512, padding=0,bias=False)#1024,11,11
        self.bn8 = nn.BatchNorm2d(1024)
        self.maxpool5 = nn.AdaptiveMaxPool2d((4,4))
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1,groups=1024, padding=0,bias=False)#2048,1,1
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.maxpool4(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.maxpool5(x)
        x = self.conv9(x)
        return x.view(-1, 1024)
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    return net

def detect_faces(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype(int)
            area = (box[2] - box[0]) * (box[3] - box[1])
            faces.append((box, area, confidence))
    return faces

def get_largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda x: x[1])

def crop_and_resize(frame, box, size=(112, 112)):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, size)
    return face

def face_to_tensor(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(face)
    return tensor

def draw_faces(frame, face_data, fps, match_info=None, face_image=None):
    if face_data is not None:
        box, area, confidence = face_data
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Area: {area}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if match_info is not None:
        cv2.putText(frame, match_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    if face_image is not None:
        h, w = frame.shape[:2]
        face_h, face_w = face_image.shape[:2]
        face_display = cv2.resize(face_image, (112, 112))
        x_offset = w - 122
        y_offset = 10
        frame[y_offset:y_offset+112, x_offset:x_offset+112] = face_display
        cv2.rectangle(frame, (x_offset, y_offset), (x_offset+112, y_offset+112), (0, 255, 0), 2)
    
    return frame

def main():
    net = load_model()
    print("模型加载成功")
    facenet=Facenet()
    facenet.load_state_dict(torch.load('best_model.pth',map_location='cpu')['model_state_dict'])
    facenet.eval()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    print("按 'q' 键退出")

    prev_time = time.time()
    fps = 0
    face_tensor = None
    listed_faces=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        faces = detect_faces(net, frame)
        largest_face = get_largest_face(faces)
        
        match_info = None
        face_image = None
        if largest_face is not None:
            box, _, _ = largest_face
            face = crop_and_resize(frame, box, (112, 112))
            face_image = face.copy()
            face_tensor = face_to_tensor(face)
            if face_tensor is not None:
                with torch.no_grad():
                    output = facenet(face_tensor.unsqueeze(0)).squeeze(0)
                    conf=[]
                    for f in listed_faces:
                        conf.append(torch.cosine_similarity(output,f, dim=0))
                if conf:
                    match_info = f"Face {conf.index(max(conf))}, conf: {max(conf):.2f}"
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        frame = draw_faces(frame, largest_face, fps, match_info, face_image)

        cv2.imshow("Face Detection", frame)
        key=cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('i'):
            if face_tensor is not None:
                listed_faces.append(output)
                print(f"Face {len(listed_faces)-1} added")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
