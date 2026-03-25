#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

const std::string PROTOTXT = "./face_detector/deploy.prototxt";
const std::string CAFFEMODEL = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel";
const std::string MODEL_PATH = "./facenet_model.pt";
const double CONFIDENCE_THRESHOLD = 0.5;

struct Face {
    cv::Rect box;
    int area;
    double confidence;
};

class FacenetFeatureExtractor {
public:
    FacenetFeatureExtractor() : device_(torch::kCPU) {}
    
    bool load_model(const std::string& model_path) {
        try {
            module_ = torch::jit::load(model_path);
            module_.eval();
            
            std::cout << "Facenet 模型加载成功:" << model_path << std::endl;
            return true;
        } catch (const c10::Error& e) {
            std::cerr << "Facenet 模型加载失败:" << e.msg() << std::endl;
            return false;
        }
    }
    
    torch::Tensor extract_feature(const cv::Mat& face) {
        cv::Mat rgb;
        cv::cvtColor(face, rgb, cv::COLOR_BGR2RGB);
        
        std::vector<float> mean = {0.5f, 0.5f, 0.5f};
        std::vector<float> std = {0.5f, 0.5f, 0.5f};
        
        torch::Tensor tensor = torch::from_blob(
            rgb.ptr<uchar>(0), 
            {112, 112, 3}, 
            torch::kUInt8
        );
        
        tensor = tensor.toType(torch::kFloat32).div(255.0);
        tensor = tensor.permute({2, 0, 1});
        
        for (int c = 0; c < 3; ++c) {
            tensor[c] = (tensor[c] - mean[c]) / std[c];
        }
        
        tensor = tensor.unsqueeze(0);
        
        std::vector<torch::jit::IValue> inputs{tensor};
        torch::Tensor output = module_.forward(inputs).toTensor();
        
        return output.squeeze(0);
    }
    
    void set_device(torch::Device device) {
        device_ = device;
        module_.to(device);
    }

private:
    torch::jit::script::Module module_;
    torch::Device device_;
};

cv::Mat blobFromImage(const cv::Mat& image) {
    return cv::dnn::blobFromImage(
        image, 
        1.0, 
        cv::Size(300, 300), 
        cv::Scalar(104.0, 177.0, 123.0),
        false,
        false
    );
}

std::vector<Face> detect_faces(cv::dnn::Net& net, const cv::Mat& frame) {
    int h = frame.rows;
    int w = frame.cols;
    
    cv::Mat input;
    cv::resize(frame, input, cv::Size(300, 300));
    cv::Mat blob = blobFromImage(input);
    net.setInput(blob);
    cv::Mat detections = net.forward();
    
    std::vector<Face> faces;
    float* data = (float*)detections.data;
    
    for (int i = 0; i < detections.size[2]; ++i) {
        double confidence = data[i * 7 + 2];
        
        if (confidence > CONFIDENCE_THRESHOLD) {
            int x1 = static_cast<int>(data[i * 7 + 3] * w);
            int y1 = static_cast<int>(data[i * 7 + 4] * h);
            int x2 = static_cast<int>(data[i * 7 + 5] * w);
            int y2 = static_cast<int>(data[i * 7 + 6] * h);
            
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(w, x2);
            y2 = std::min(h, y2);
            
            cv::Rect box(x1, y1, x2 - x1, y2 - y1);
            int area = box.width * box.height;
            
            faces.push_back({box, area, confidence});
        }
    }
    
    return faces;
}

Face* get_largest_face(std::vector<Face>& faces) {
    if (faces.empty()) {
        return nullptr;
    }
    
    return &*std::max_element(faces.begin(), faces.end(), 
        [](const Face& a, const Face& b) {
            return a.area < b.area;
        });
}

cv::Mat crop_and_resize(const cv::Mat& frame, const cv::Rect& box, cv::Size size = cv::Size(112, 112)) {
    cv::Rect clipped_box = box & cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat face = frame(clipped_box);
    cv::Mat resized;
    cv::resize(face, resized, size);
    return resized;
}

double Cosine_Similarity(const torch::Tensor& a, const torch::Tensor& b) {
    torch::Tensor dot_product = torch::sum(a * b);
    double norm_a = torch::sqrt(torch::sum(a * a)).item<double>();
    double norm_b = torch::sqrt(torch::sum(b * b)).item<double>();
    
    if (norm_a == 0 || norm_b == 0) {
        return 0.0;
    }
    
    return dot_product.item<double>() / (norm_a * norm_b);
}

void draw_faces(cv::Mat& frame, Face* face_data, double fps, 
                const std::string& match_info = "", 
                const cv::Mat& face_image = cv::Mat()) {
    if (face_data != nullptr) {
        cv::rectangle(frame, face_data->box, cv::Scalar(0, 255, 0), 2);
        std::string area_text = "Area: " + std::to_string(face_data->area);
        cv::putText(frame, area_text, 
                   cv::Point(face_data->box.x, face_data->box.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    
    std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
    cv::putText(frame, fps_text, cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    
    if (!match_info.empty()) {
        cv::putText(frame, match_info, cv::Point(10, 70), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
    
    if (!face_image.empty()) {
        cv::Mat face_display;
        cv::resize(face_image, face_display, cv::Size(112, 112));
        
        int x_offset = frame.cols - 122;
        int y_offset = 10;
        
        cv::Mat roi = frame(cv::Rect(x_offset, y_offset, 112, 112));
        face_display.copyTo(roi);
        
        cv::rectangle(frame, cv::Rect(x_offset, y_offset, 112, 112), 
                     cv::Scalar(0, 255, 0), 2);
    }
}

int main() {
    torch::NoGradGuard no_grad;
    
    FacenetFeatureExtractor facenet;
    if (!facenet.load_model(MODEL_PATH)) {
        std::cerr << "无法加载 Facenet 模型" << std::endl;
        return -1;
    }
    
    cv::dnn::Net face_net;
    try {
        face_net = cv::dnn::readNetFromCaffe(PROTOTXT, CAFFEMODEL);
        std::cout << "人脸检测模型加载成功." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "人脸检测模型加载失败:" << e.what() << std::endl;
        return -1;
    }
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头." << std::endl;
        return -1;
    }
    
    cv::namedWindow("Face Detection", cv::WINDOW_NORMAL);
    
    std::cout << "按 'q' 键退出，按 'i' 键添加当前人脸到数据库." << std::endl;
    
    auto prev_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    
    torch::Tensor current_tensor;
    std::vector<torch::Tensor> listed_faces;
    
    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "无法读取帧." << std::endl;
            break;
        }
        
        std::vector<Face> faces = detect_faces(face_net, frame);
        Face* largest_face = get_largest_face(faces);
        
        std::string match_info;
        cv::Mat face_image;
        
        if (largest_face != nullptr) {
            cv::Mat face = crop_and_resize(frame, largest_face->box, cv::Size(112, 112));
            face_image = face.clone();
            
            current_tensor = facenet.extract_feature(face);
            
            if (!listed_faces.empty()) {
                std::vector<double> confidences;
                for (const auto& stored_face : listed_faces) {
                    double sim = Cosine_Similarity(current_tensor, stored_face);
                    confidences.push_back(sim);
                }
                
                auto max_it = std::max_element(confidences.begin(), confidences.end());
                int max_idx = static_cast<int>(std::distance(confidences.begin(), max_it));
                double max_conf = *max_it;
                if (max_conf > 0.6)
                    match_info = "Face " + std::to_string(max_idx) + ", conf: " + std::to_string(static_cast<int>(max_conf * 100) / 100.0);
                else
                    match_info = "Face Unknown";
            }
        }
        
        auto curr_time = std::chrono::high_resolution_clock::now();
        fps = 1.0 / std::chrono::duration<double>(curr_time - prev_time).count();
        prev_time = curr_time;
        
        draw_faces(frame, largest_face, fps, match_info, face_image);
        
        cv::imshow("Face Detection", frame);
        
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == 'i' && current_tensor.numel() > 0) {
            listed_faces.push_back(current_tensor.clone());
            std::cout << "Face " << (listed_faces.size() - 1) << " added" << std::endl;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
