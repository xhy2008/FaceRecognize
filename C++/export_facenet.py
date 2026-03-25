import torch
import torch.nn as nn
import torch.quantization

class Facenet(nn.Module):
    def __init__(self):
        super(Facenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.AdaptiveMaxPool2d((55, 55))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.AdaptiveMaxPool2d((27, 27))
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, groups=64, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, groups=128, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.AdaptiveMaxPool2d((13, 13))
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, groups=256, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, groups=512, padding=0, bias=False)
        self.bn8 = nn.BatchNorm2d(1024)
        self.maxpool5 = nn.AdaptiveMaxPool2d((4, 4))
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1, groups=1024, padding=0, bias=False)

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
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.maxpool5(x)
        x = self.conv9(x)
        return x.view(-1, 1024)

def export_model(weights_path='best_model.pth', output_path='facenet_model.pt'):
    print("正在加载 Facenet 模型...")
    model = Facenet()
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("原始模型大小:", sum(p.numel() for p in model.parameters()), "参数")
    
    print("\n应用 int8 动态量化...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    
    print("量化后模型大小:", sum(p.numel() for p in quantized_model.parameters()), "参数")
    
    dummy_input = torch.randn(1, 3, 112, 112)
    
    print("\n导出量化模型到 TorchScript...")
    traced_model = torch.jit.trace(quantized_model, dummy_input)
    traced_model.save(output_path)
    
    print(f"\n模型已成功导出到：{output_path}")
    print("该模型已进行 int8 量化，适用于 CPU 推理")
    
    test_output = traced_model(dummy_input)
    print(f"测试输出形状：{test_output.shape}")

if __name__ == "__main__":
    export_model()
