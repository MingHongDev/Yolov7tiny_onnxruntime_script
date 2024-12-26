#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "OnnxInferenceBase.hpp"
#include "Yolov7.hpp"

using namespace cv;
using namespace std;

const Size netInputSize(256, 256);
const string inputImagePath = "./data/0.png";
const string ODClassFilePath = "./data/OD_classes.txt";
const string modelPath = "./data/QInt8_model.onnx";

vector<string> input_node_names = { "images" };
vector<int64_t> input_dims = { 1, 3, netInputSize.height, netInputSize.width};
vector<string> output_node_names = { "265", "274", "output" };

int main(){
    //Set model
    Yolov7 model;
    model.initDetector(0.45, 0.45, 0.45, ODClassFilePath);
    model.setSessionOptions(false);
    model.loadWeights(modelPath);
    model.setInputDemensions(input_dims);
    model.setInputNodeNames(input_node_names);
    model.setOutputNodeNames(output_node_names);

    //Read image
    Mat src = imread(inputImagePath);
    Mat netInput;
    resize(src, netInput, netInputSize, 0, 0, INTER_NEAREST);

    //Inference
    model.inference(netInput);
    model.draw(src);
    imshow("Result", src);
    waitKey(0);

    return 0;
}