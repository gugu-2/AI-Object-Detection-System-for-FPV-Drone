#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

class FPVObjectDetector {
private:
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    cv::VideoCapture cap;
    
    // Detection parameters
    float confThreshold = 0.5f;
    float nmsThreshold = 0.4f;
    int inputWidth = 416;
    int inputHeight = 416;
    
    // FPV specific parameters
    bool lowLatencyMode = true;
    int frameSkip = 0; // Skip frames for better performance
    
public:
    struct Detection {
        int classId;
        float confidence;
        cv::Rect box;
        std::string className;
    };
    
    struct DroneTarget {
        cv::Point2f center;
        float distance;
        float bearing;
        std::string type;
        float confidence;
    };

    FPVObjectDetector() {}
    
    bool initializeModel(const std::string& modelPath, const std::string& configPath, const std::string& classNamesPath) {
        try {
            // Load YOLO network
            net = cv::dnn::readNetFromDarknet(configPath, modelPath);
            
            // Set backend and target for optimization
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Change to DNN_TARGET_CUDA for GPU
            
            // Load class names
            loadClassNames(classNamesPath);
            
            std::cout << "Model initialized successfully!" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error initializing model: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool initializeCamera(int cameraIndex = 0) {
        cap.open(cameraIndex);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera " << cameraIndex << std::endl;
            return false;
        }
        
        // Set camera properties for FPV optimization
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // Minimize buffer for low latency
        
        std::cout << "Camera initialized successfully!" << std::endl;
        return true;
    }
    
    void loadClassNames(const std::string& classNamesPath) {
        std::ifstream ifs(classNamesPath);
        std::string line;
        while (std::getline(ifs, line)) {
            classNames.push_back(line);
        }
    }
    
    std::vector<Detection> detectObjects(const cv::Mat& frame) {
        std::vector<Detection> detections;
        
        // Create blob from frame
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(0,0,0), true, false);
        net.setInput(blob);
        
        // Run forward pass
        std::vector<cv::Mat> outs;
        net.forward(outs, getOutputNames());
        
        // Post-process detections
        postProcess(frame, outs, detections);
        
        return detections;
    }
    
    std::vector<DroneTarget> analyzeForDrone(const std::vector<Detection>& detections, const cv::Mat& frame) {
        std::vector<DroneTarget> targets;
        
        for (const auto& detection : detections) {
            DroneTarget target;
            target.center = cv::Point2f(detection.box.x + detection.box.width/2.0f, 
                                       detection.box.y + detection.box.height/2.0f);
            target.type = detection.className;
            target.confidence = detection.confidence;
            
            // Calculate relative position from frame center
            cv::Point2f frameCenter(frame.cols/2.0f, frame.rows/2.0f);
            cv::Point2f relative = target.center - frameCenter;
            
            // Estimate bearing (simplified)
            target.bearing = atan2(relative.y, relative.x) * 180.0f / CV_PI;
            
            // Estimate distance based on object size (simplified)
            float objectArea = detection.box.width * detection.box.height;
            target.distance = estimateDistance(detection.className, objectArea);
            
            targets.push_back(target);
        }
        
        return targets;
    }
    
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections) {
        for (const auto& detection : detections) {
            // Draw bounding box
            cv::rectangle(frame, detection.box, cv::Scalar(0, 255, 0), 2);
            
            // Draw label
            std::string label = detection.className + ": " + 
                               cv::format("%.2f", detection.confidence);
            
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            
            cv::rectangle(frame, 
                         cv::Point(detection.box.x, detection.box.y - labelSize.height - 10),
                         cv::Point(detection.box.x + labelSize.width, detection.box.y),
                         cv::Scalar(0, 255, 0), -1);
            
            cv::putText(frame, label, 
                       cv::Point(detection.box.x, detection.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    }
    
    void drawDroneHUD(cv::Mat& frame, const std::vector<DroneTarget>& targets) {
        // Draw crosshair
        cv::Point2f center(frame.cols/2.0f, frame.rows/2.0f);
        cv::line(frame, cv::Point(center.x - 20, center.y), cv::Point(center.x + 20, center.y), cv::Scalar(255, 255, 255), 2);
        cv::line(frame, cv::Point(center.x, center.y - 20), cv::Point(center.x, center.y + 20), cv::Scalar(255, 255, 255), 2);
        
        // Draw target information
        int yOffset = 30;
        for (const auto& target : targets) {
            std::string targetInfo = target.type + " | Dist: " + 
                                   cv::format("%.1fm", target.distance) + 
                                   " | Bear: " + cv::format("%.1f°", target.bearing);
            
            cv::putText(frame, targetInfo, cv::Point(10, yOffset), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            yOffset += 25;
            
            // Draw target indicator
            cv::circle(frame, target.center, 5, cv::Scalar(0, 0, 255), -1);
            cv::circle(frame, target.center, 15, cv::Scalar(0, 0, 255), 2);
        }
        
        // Display FPS
        static auto lastTime = std::chrono::high_resolution_clock::now();
        static int frameCount = 0;
        static float fps = 0.0f;
        
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
        
        if (duration.count() >= 1000) {
            fps = frameCount * 1000.0f / duration.count();
            frameCount = 0;
            lastTime = currentTime;
        }
        
        cv::putText(frame, "FPS: " + cv::format("%.1f", fps), 
                   cv::Point(frame.cols - 100, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
    
    void run() {
        if (!cap.isOpened()) {
            std::cerr << "Camera not initialized!" << std::endl;
            return;
        }
        
        cv::Mat frame;
        int frameCounter = 0;
        
        std::cout << "Starting FPV Object Detection..." << std::endl;
        std::cout << "Press 'q' to quit, 's' to save screenshot" << std::endl;
        
        while (true) {
            cap >> frame;
            if (frame.empty()) break;
            
            // Skip frames for performance if needed
            if (lowLatencyMode && frameCounter % (frameSkip + 1) != 0) {
                frameCounter++;
                continue;
            }
            
            // Detect objects
            auto detections = detectObjects(frame);
            
            // Analyze for drone-specific targets
            auto targets = analyzeForDrone(detections, frame);
            
            // Draw visualizations
            drawDetections(frame, detections);
            drawDroneHUD(frame, targets);
            
            // Display frame
            cv::imshow("FPV Object Detection", frame);
            
            // Handle keyboard input
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;
            if (key == 's') {
                std::string filename = "fpv_screenshot_" + 
                                     std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                                         std::chrono::system_clock::now().time_since_epoch()).count()) + ".jpg";
                cv::imwrite(filename, frame);
                std::cout << "Screenshot saved: " << filename << std::endl;
            }
            if (key == 'p') {
                lowLatencyMode = !lowLatencyMode;
                std::cout << "Low latency mode: " << (lowLatencyMode ? "ON" : "OFF") << std::endl;
            }
            
            frameCounter++;
        }
        
        cap.release();
        cv::destroyAllWindows();
    }
    
    void setDetectionParameters(float confThresh, float nmsThresh) {
        confThreshold = confThresh;
        nmsThreshold = nmsThresh;
    }
    
    void setPerformanceMode(bool lowLatency, int skipFrames = 0) {
        lowLatencyMode = lowLatency;
        frameSkip = skipFrames;
    }

private:
    std::vector<std::string> getOutputNames() {
        static std::vector<std::string> names;
        if (names.empty()) {
            std::vector<int> outLayers = net.getUnconnectedOutLayers();
            std::vector<std::string> layersNames = net.getLayerNames();
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i) {
                names[i] = layersNames[outLayers[i] - 1];
            }
        }
        return names;
    }
    
    void postProcess(const cv::Mat& frame, const std::vector<cv::Mat>& outs, std::vector<Detection>& detections) {
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        
        for (size_t i = 0; i < outs.size(); ++i) {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }
        
        // Apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Detection detection;
            detection.classId = classIds[idx];
            detection.confidence = confidences[idx];
            detection.box = boxes[idx];
            detection.className = (detection.classId < classNames.size()) ? 
                                 classNames[detection.classId] : "Unknown";
            detections.push_back(detection);
        }
    }
    
    float estimateDistance(const std::string& className, float objectArea) {
        // Simplified distance estimation based on object type and apparent size
        // This would need calibration with real-world measurements
        std::map<std::string, float> referenceAreas = {
            {"person", 10000.0f},
            {"car", 15000.0f},
            {"truck", 25000.0f},
            {"building", 50000.0f}
        };
        
        auto it = referenceAreas.find(className);
        if (it != referenceAreas.end()) {
            return sqrt(it->second / objectArea) * 10.0f; // Rough estimation
        }
        
        return sqrt(10000.0f / objectArea) * 10.0f; // Default estimation
    }
};

// Usage example and main function
int main() {
    FPVObjectDetector detector;
    
    // Initialize the model (you'll need to download YOLO weights, config, and class names)
    std::string modelPath = "yolov4.weights";          // Download from YOLO repository
    std::string configPath = "yolov4.cfg";             // Download from YOLO repository  
    std::string classNamesPath = "coco.names";         // COCO class names file
    
    if (!detector.initializeModel(modelPath, configPath, classNamesPath)) {
        std::cerr << "Failed to initialize model. Make sure you have:" << std::endl;
        std::cerr << "1. yolov4.weights file" << std::endl;
        std::cerr << "2. yolov4.cfg file" << std::endl;
        std::cerr << "3. coco.names file" << std::endl;
        std::cerr << "Download from: https://github.com/AlexeyAB/darknet" << std::endl;
        return -1;
    }
    
    // Initialize camera
    if (!detector.initializeCamera(0)) { // Use camera index 0
        return -1;
    }
    
    // Set detection parameters for FPV use
    detector.setDetectionParameters(0.3f, 0.4f); // Lower confidence for more detections
    detector.setPerformanceMode(true, 1); // Low latency mode, skip 1 frame
    
    // Run the detection system
    detector.run();
    
    return 0;
}