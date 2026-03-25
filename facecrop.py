import cv2
import numpy as np
import os
import argparse
import glob

# 支持的图片格式
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

# 加载Caffe模型
def load_caffe_model():
    # 模型文件路径
    model_file = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "face_detector/deploy.prototxt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_file) or not os.path.exists(config_file):
        print("错误：找不到模型文件，请确保face_detector目录下有以下文件：")
        print("- res10_300x300_ssd_iter_140000.caffemodel")
        print("- deploy.prototxt")
        return None, None
    
    # 加载模型
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

# 检测人脸
def detect_faces(net, image, confidence_threshold=0.7):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    
    # 构建一个blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # 输入blob到网络
    net.setInput(blob)
    
    # 人脸检测
    detections = net.forward()
    faces = []
    
    # 遍历检测结果
    for i in range(0, detections.shape[2]):
        # 获取置信度
        confidence = detections[0, 0, i, 2]
        
        # 过滤低置信度的检测
        if confidence > confidence_threshold:
            # 计算人脸边界框的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # 确保边界框在图像范围内
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)
            
            # 计算人脸面积
            face_area = (endX - startX) * (endY - startY)
            
            # 添加到人脸列表
            faces.append({
                'box': (startX, startY, endX, endY),
                'confidence': confidence,
                'area': face_area
            })
    
    return faces

# 裁剪并调整最大人脸
def crop_and_resize_largest_face(image, faces, target_size=(128, 128)):
    if not faces:
        print("警告：未检测到人脸")
        return None
    
    # 找到面积最大的人脸
    largest_face = max(faces, key=lambda x: x['area'])
    (startX, startY, endX, endY) = largest_face['box']
    
    # 裁剪人脸
    face_roi = image[startY:endY, startX:endX]
    
    # 调整大小
    resized_face = cv2.resize(face_roi, target_size)
    
    return resized_face

# 处理单张图片
def process_single_image(net, image_path, confidence_threshold=0.7):
    print(f"处理图片: {image_path}")
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return False
    
    # 检测人脸
    faces = detect_faces(net, image, confidence_threshold)
    
    if not faces:
        print(f"未检测到人脸: {image_path}")
        return False
    
    print(f"检测到 {len(faces)} 张人脸")
    
    # 裁剪并调整最大人脸
    cropped_face = crop_and_resize_largest_face(image, faces)
    
    if cropped_face is None:
        print(f"裁剪人脸失败: {image_path}")
        return False
    
    # 直接覆盖原文件
    cv2.imwrite(image_path, cropped_face)
    print(f"已保存处理结果到: {image_path}")
    return True

# 批量处理目录中的所有图片
def process_directory(net, directory, confidence_threshold=0.7):
    print(f"开始批量处理目录: {directory}")
    
    # 获取所有支持的图片文件
    image_files = []
    for extension in IMAGE_EXTENSIONS:
        # 查找当前目录及所有子目录中的图片
        pattern = os.path.join(directory, "**", extension)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    if not image_files:
        print("未找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片文件
    success_count = 0
    for image_path in image_files:
        try:
            if process_single_image(net, image_path, confidence_threshold):
                success_count += 1
        except Exception as e:
            print(f"处理图片时出错 {image_path}: {str(e)}")
    
    print(f"批量处理完成。成功处理 {success_count}/{len(image_files)} 个文件")

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='人脸检测和裁剪工具')
    parser.add_argument('--input', '-i', required=True, help='输入图片路径或目录路径')
    parser.add_argument('--output', '-o', help='输出图片路径（仅在处理单张图片且不希望覆盖原文件时使用）')
    parser.add_argument('--confidence', '-c', type=float, default=0.7, 
                        help='人脸检测置信度阈值（默认：0.7）')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='是否批量处理目录中的所有图片')
    args = parser.parse_args()
    
    # 加载模型
    print("加载Caffe模型...")
    net = load_caffe_model()
    if net is None:
        return
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单张图片
        if args.batch:
            print("警告：指定了--batch参数但输入是文件，将按单文件模式处理")
        
        # 检查输出路径设置
        if args.output is None:
            # 默认覆盖原文件
            output_path = args.input
        else:
            output_path = args.output
        
        print(f"读取图片: {args.input}")
        image = cv2.imread(args.input)
        if image is None:
            print("错误：无法读取图片")
            return
        
        # 检测人脸
        print("检测人脸...")
        faces = detect_faces(net, image, args.confidence)
        
        if not faces:
            print("未检测到人脸")
            return
        
        print(f"检测到 {len(faces)} 张人脸")
        
        # 裁剪并调整最大人脸
        print("裁剪并调整最大人脸...")
        cropped_face = crop_and_resize_largest_face(image, faces)
        
        if cropped_face is None:
            print("裁剪人脸失败")
            return
        
        # 保存结果
        print(f"保存结果到: {output_path}")
        cv2.imwrite(output_path, cropped_face)
        
        # 显示结果
        cv2.imshow("Original Image", image)
        cv2.imshow("Cropped Face", cropped_face)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif os.path.isdir(args.input):
        # 批量处理目录
        if args.output:
            print("警告：处理目录时忽略--output参数")
        
        process_directory(net, args.input, args.confidence)
    else:
        print(f"错误：输入路径不存在 {args.input}")
        return

if __name__ == "__main__":
    main()