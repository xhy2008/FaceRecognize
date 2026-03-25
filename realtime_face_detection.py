import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
PROTOTXT = "face_detector/deploy.prototxt"
CAFFEMODEL = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
USENET="CONVNET"
CONFIDENCE_THRESHOLD = 0.5

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #input:3*112*112
        #空洞卷积
        self.conv1=nn.Conv2d(3,8,kernel_size=3,stride=1,padding=0,bias=False)#8*110*110
        self.bn1=nn.BatchNorm2d(8)
        self.relu1=nn.ReLU(inplace=True)
        self.convlayer1=nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=1,padding=0,bias=False),#16*108*108
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=0,bias=False),#32*106*106 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0,bias=False),#64*104*104
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1=nn.AdaptiveMaxPool2d((50,50))#64*50*50
        #深度可分离卷积
        self.convlayer2=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=0,groups=64,bias=False),#128*48*48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=0,groups=128,bias=False),#256*46*46
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=0,groups=256,bias=False),#512*44*44
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=0,groups=512,bias=False),#1024*42*42
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.pool2=nn.AdaptiveMaxPool2d((20,20))#1024*20*20
        #线性卷积
        self.convlayer3=nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=0,bias=False),#512*18*18
            nn.BatchNorm2d(512),
            nn.Conv2d(512,256,kernel_size=3,stride=1,padding=0,bias=False),#256*16*16
            nn.BatchNorm2d(256),
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=0,bias=False),#128*14*14
            nn.BatchNorm2d(128),
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=0,bias=False),#64*14*14
            nn.BatchNorm2d(64),
        )
        self.pool3=nn.AdaptiveAvgPool2d((6,6))#64*6*6
        self.linear=nn.Conv2d(64,32,kernel_size=3,stride=1,padding=0,bias=False)#32*4*4

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.convlayer1(x)
        x=self.pool1(x)
        x=self.convlayer2(x)
        x=self.pool2(x)
        x=self.convlayer3(x)
        x=self.pool3(x)
        x=self.linear(x)
        x=x.view(x.size(0),-1)#batch*1024
        return x

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            #pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]
class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x

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
    if USENET=="MOBILEFACENET":
        convnet=MobileFacenet()
        convnet.load_state_dict(torch.load('mobilefacenet.ckpt',map_location='cpu')["net_state_dict"])
    elif USENET=="CONVNET":
        convnet=ConvNet()
        convnet.load_state_dict(torch.load('best_model_origin.pth',map_location='cpu')['model_state_dict'])
    elif USENET=="MOBILEFACENET_FINETUNE":
        convnet=MobileFacenet()
        convnet.load_state_dict(torch.load('mobilefacenet_finetuned.ckpt',map_location='cpu')['backbone_state_dict'])
    elif USENET=="FACENET":
        from modelv2 import Facenet
        convnet=Facenet()
        convnet.load_state_dict(torch.load('best_model.pth',map_location='cpu')['model_state_dict'])
    convnet.eval()
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
            box, area, confidence = largest_face
            face = crop_and_resize(frame, box, (112, 112))
            face_image = face.copy()
            face_tensor = face_to_tensor(face)
            if face_tensor is not None:
                with torch.no_grad():
                    output = convnet(face_tensor.unsqueeze(0)).squeeze(0)
                    conf=[]
                    for f in listed_faces:
                        conf.append(torch.cosine_similarity(output,f, dim=0))
                if conf:
                    match_info = f"Face {conf.index(max(conf))}, conf: {max(conf):.2f}"
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        frame = draw_faces(frame, largest_face, fps*5, match_info, face_image)

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
