//
//  Yolov7.hpp
//
//  Created by MH on 2024/12/24.
//

#ifndef Yolov7_hpp
#define Yolov7_hpp

#include <stdio.h>
#include <math.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "OnnxInferenceBase.hpp"

class Yolov7 : public OnnxInferenceBase {
public:
    Yolov7() {};
    ~Yolov7() {};
    
    std::vector<cv::Rect> bboxes;
    std::vector<std::string> labeles;
    std::vector<float> confes;
    
    void initDetector(float confThreshold, float nmsThreshold, float boxhreshold, std::string classesFile);
    void inference(cv::Mat& frame);
    void draw(cv::Mat& img);
    
private:
    const bool debugMode = false;
    const int modelInputShape[2] = {256, 256};
    const float netStride[3] = {8.0, 16.0, 32.0};
    const float netAnchors[3][6] = {{3.0, 11.0, 6.0, 23.0, 9.0, 30.0}, {15, 47, 21, 77, 37, 68}, {43, 134, 88, 145, 208, 219}};
    
    float nmsThreshold;
    float boxThreshold;
    float probThreshold;

    std::vector<std::string> classes;
    int numClass;

    void decodeBox(std::vector<cv::Mat>& modelOuts);

    float Sigmoid(float x) {
        return static_cast<float> (1.0 / (1.0 + exp(-x)));
    }

};

#endif 
