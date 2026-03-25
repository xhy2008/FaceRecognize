import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from model import ConvNet
from mobilefacenet import MobileFacenet
USENET="CONVNET"
if USENET=="MOBILEFACENET":
    convnet=MobileFacenet().to('cuda')
    convnet.load_state_dict(torch.load('mobilefacenet.ckpt',map_location='cuda')["net_state_dict"])
elif USENET=="CONVNET":
    convnet=ConvNet().to('cuda')
    convnet.load_state_dict(torch.load('best_model.pth',map_location='cuda')['model_state_dict'])
elif USENET=="MOBILEFACENET_FINETUNE":
    convnet=MobileFacenet().to('cuda')
    convnet.load_state_dict(torch.load('mobilefacenet_finetuned.ckpt',map_location='cuda')['backbone_state_dict'])
convnet.eval()
# 加载两张图像
img1 = Image.open('image1.jpg').convert('RGB')
img2 = Image.open('image2.jpg').convert('RGB')

# 预处理
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
tensor1 = transform(img1).unsqueeze(0).to('cuda')
tensor2 = transform(img2).unsqueeze(0).to('cuda')

# 提取特征
with torch.no_grad():
    feat1 = convnet(tensor1)
    feat2 = convnet(tensor2)

# 计算余弦相似度
cos_sim = torch.nn.functional.cosine_similarity(feat1, feat2, dim=1).item()
print(f"两张图像的相似度: {cos_sim:.4f}")
