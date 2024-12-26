//
//  Yolov7.cpp
//  
//  Created by MH on 2024/12/24.
//
#include <fstream>
#include <sstream>
#include <iostream>
#include "Yolov7.hpp"

using namespace cv;
using namespace dnn;
using namespace std;


void Yolov7::initDetector(float confThreshold, float nmsThreshold, float boxThreshold, string classesFile) {
    this->nmsThreshold = nmsThreshold;
    this->boxThreshold = boxThreshold;
    this->probThreshold = confThreshold;

    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) {
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(remove(line.begin(), line.end(), '\t'), line.end());
        line.erase(remove(line.begin(), line.end(), '\0'), line.end());
        classes.push_back(line);
    }
    numClass = int(classes.size());
}

void Yolov7::decodeBox(vector<Mat>& modelOuts) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    int netWidth = numClass + 5;
    const float ratioH = 1.0f; 
    const float ratioW = 1.0f;
    
    int strideIdx = 0;
    for (int stride = 0; stride < modelOuts.size(); stride++) {
        if (modelOuts.size() == 1) {
            strideIdx++;
        } else {
            strideIdx = stride;
        }
        int gridX = (int)(modelInputShape[1] / netStride[strideIdx]);
        int gridY = (int)(modelInputShape[0] / netStride[strideIdx]);
        float* pdata = (float*)modelOuts[stride].data;
        
        for(int i = 0; i < gridY; i++) {
            for(int j = 0; j < gridX; j++) {
                for (int anchor = 0; anchor < 3; anchor++) {
                    float anchorW = netAnchors[strideIdx][anchor * 2];
                    float anchorH = netAnchors[strideIdx][anchor * 2 + 1];
                    float boxScore = Sigmoid(pdata[4]);
                    if (boxScore >= boxThreshold) {
                        Mat scores(1, numClass, CV_32FC1, pdata + 5);
                        Point classIdPoint;
						double maxClassScore;

                        minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);                        
						maxClassScore = Sigmoid((float)maxClassScore);

                        if (maxClassScore >= probThreshold) {
							float x = (Sigmoid(pdata[0]) * 2.0 - 0.5 + j) * netStride[strideIdx];
							float y = (Sigmoid(pdata[1]) * 2.0 - 0.5 + i) * netStride[strideIdx];
							float w = powf(Sigmoid(pdata[2]) * 2.0, 2.0) * anchorW;
							float h = powf(Sigmoid(pdata[3]) * 2.0, 2.0) * anchorH;
							int left = (int)(x - 0.5 * w) * ratioW + 0.5;
							int top = (int)(y - 0.5 * h) * ratioH + 0.5;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(maxClassScore * boxScore);
							boxes.push_back(Rect(left, top, int(w * ratioW), int(h * ratioH)));
						}
                    }
                    pdata += netWidth;
                }
            }
        }
    }

    vector<int> nmsResult;
	NMSBoxes(boxes, confidences, probThreshold, nmsThreshold, nmsResult);

    bboxes = vector<Rect>();
    labeles = vector<string>();
    confes = vector<float>();
    for (int i = 0; i < nmsResult.size(); i++) {
        int idx = nmsResult[i];
        if (confidences[idx] > probThreshold) {
            labeles.push_back(classes[classIds[idx]]);
            bboxes.push_back(boxes[idx]);
            confes.push_back(confidences[idx]);
        }
    }
}

void Yolov7::draw(Mat& img){
    float ratioH = static_cast<float>(img.rows) / modelInputShape[0];
    float ratioW = static_cast<float>(img.cols) / modelInputShape[1];
    for (int k = 0; k < bboxes.size(); k++) {
        auto bbox = bboxes[k];
        if(confes[k] > 0) {
            string drawLabel = labeles[k];
            drawLabel = drawLabel + format(" : %.2f", confes[k]);
            if (bbox.width > 0 && bbox.height > 0) {
                Rect scaledBox = Rect(bbox.x*ratioW, bbox.y*ratioH, bbox.width*ratioW, bbox.height*ratioH);
                int baseLine;
                Size labelSize = getTextSize(drawLabel, FONT_HERSHEY_SIMPLEX, 0.75, 2, &baseLine);
                rectangle(img, scaledBox, Scalar(0, 0, 255), 11);
                putText(img, drawLabel, Point(scaledBox.x, max(scaledBox.y - labelSize.height + 2, 0)), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 255), 2);
            }
        }
    }
}

void printTensorMatShape(const Mat& mat) {
    cout << "Dims: " << mat.dims << ", Shape: ";
    for (int i = 0; i < mat.dims; ++i) {
        cout << mat.size[i];
        if (i < mat.dims - 1) cout << " x ";
    }
    cout << endl;
}

void Yolov7::inference(Mat& BGR) {
    const Size inputSize(modelInputShape[0], modelInputShape[1]);
    if(BGR.size() != inputSize) {
        cout << "error!" << endl;
        return;
    }

    // step1 : handle input
    Mat blob;
    blobFromImage(BGR, blob, 1.0 , inputSize, Scalar(0, 0, 0), true, false);
	vector<Ort::Value> inputTensor;
	try {
		inputTensor.emplace_back(
			Ort::Value::CreateTensor<float>(
				memory_info, 
				(float*)blob.data, 
				blob.total(), // totla size : modelInputShape[0] * modelInputShape[1] * 3
				input_node_dims.data(), 
				input_node_dims.size()));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return;
	}

    // step2 : handle output
    vector<Ort::Value> OutputTensor;
	try {
		OutputTensor = session.Run(Ort::RunOptions{nullptr}, 
                                    input_node_names.data(), 
                                    inputTensor.data(), 
                                    inputTensor.size(),
                                    output_node_names.data(),
                                    output_node_names.size());
	}
	catch (Ort::Exception oe) {
		cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return;
	}

    vector<Mat> onnxOuts;    
	for (size_t i = 0; i < OutputTensor.size(); ++i) {
        vector<int64_t> shape = OutputTensor[i].GetTensorTypeAndShapeInfo().GetShape();
        auto type = OutputTensor[i].GetTensorTypeAndShapeInfo().GetElementType();
        vector<int> matSizes = {static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2])}; // B, C, H*W
        onnxOuts.push_back(Mat(matSizes, CV_32F, OutputTensor[i].GetTensorMutableData<float>()));
        
        if (debugMode) {
            printTensorShape(OutputTensor[i]);
            printTensorMatShape(onnxOuts[i]);
        }
	}

    for (int i = 0; i < onnxOuts.size(); i++) {
        if (onnxOuts[i].dims == 3) {        
            const int num_proposal = onnxOuts[i].size[1];
            onnxOuts[i] = onnxOuts[i].reshape(0, num_proposal);
            onnxOuts[i] = onnxOuts[i].t();
        }
    }

    decodeBox(onnxOuts);
}

