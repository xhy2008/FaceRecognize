import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vit import ViTFeatureExtractor

def export_model(model_path='best_vit_model.pth', output_path='C++/best_vit_model_traced.pt'):
    print("加载 ViT 模型...")
    
    device = torch.device('cpu')
    
    vit = ViTFeatureExtractor(pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)['vit_state_dict']
    vit.load_state_dict(state_dict)
    vit.eval()
    
    print("模型结构:")
    print(vit)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("\n测试模型...")
    with torch.no_grad():
        output = vit(dummy_input)
        print(f"输出形状：{output.shape}")
    
    print("追踪模型...")
    vit.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(vit, dummy_input)
    
    print(f"保存模型到 {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model.save(output_path)
    
    print("模型导出完成!")
    
    print("\n验证保存的模型...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    
    with torch.no_grad():
        test_output = loaded_model(dummy_input)
        print(f"加载模型输出形状：{test_output.shape}")
    
    print("所有步骤完成!")

if __name__ == '__main__':
    export_model()
