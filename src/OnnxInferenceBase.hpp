#pragma once
#ifndef OnnxInferenceBase_hpp
#define OnnxInferenceBase_hpp

#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>


class OnnxInferenceBase {
public:
	void setSessionOptions(bool UseCuda);
	bool loadWeights(const std::string& ModelPath);
	void setInputNodeNames(std::vector<std::string> input_node_names);
	void setInputDemensions(std::vector<int64_t> input_node_dims);
	void setOutputNodeNames(std::vector<std::string> input_node_names);
	void getNodeNames(bool is_input);

	static void printTensorShape(const Ort::Value& tensor);

protected:
	//Ort::Env need to init first.
	Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
	Ort::Session session = Ort::Session(nullptr);
	Ort::SessionOptions sessionOptions;
	OrtCUDAProviderOptions cuda_options;
	Ort::MemoryInfo memory_info{ nullptr };					// Used to allocate memory for input
	std::vector<const char*> output_node_names;				// output node names (pointer format)
	std::vector<const char*> input_node_names;				// Input node names (pointer format)
	std::vector<int64_t> input_node_dims;					// Input node dimension

private:
	std::vector<std::string> actual_output_node_names;	// output node names
	std::vector<std::string> actual_input_node_names;	// Input node names
};

#endif 