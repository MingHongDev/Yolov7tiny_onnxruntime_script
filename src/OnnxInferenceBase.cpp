#include "OnnxInferenceBase.hpp"

void OnnxInferenceBase::setSessionOptions(bool useCUDA) {
	//check CPU or GPU device
    auto providers = Ort::GetAvailableProviders();
    for (auto provider : providers) {
        std::cout << provider << std::endl;
    }

	sessionOptions.SetInterOpNumThreads(3);
	// Optimization will take time and memory during startup
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	// CUDA options. If used.
	if (useCUDA) {
		cuda_options.device_id = 0;  //GPU_ID
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
		cuda_options.arena_extend_strategy = 0;
		// May cause data race in some condition
		cuda_options.do_copy_in_default_stream = 0;
		sessionOptions.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
	}
}

bool OnnxInferenceBase::loadWeights(const std::string& ModelPath) {
	try {
		session = Ort::Session(env, ModelPath.c_str(), sessionOptions);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	try {	// For allocating memory for input tensors
		memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	return true;
}

void OnnxInferenceBase::setInputNodeNames(std::vector<std::string> names) {
	actual_input_node_names = names;
	input_node_names.clear();
	for(auto& name : actual_input_node_names) {
		input_node_names.push_back(name.c_str());
	}
}

void OnnxInferenceBase::setOutputNodeNames(std::vector<std::string> names) {
	actual_output_node_names = names;
	output_node_names.clear();
	for (auto& name : actual_output_node_names) {
		output_node_names.push_back(name.c_str());
	}
}

void OnnxInferenceBase::setInputDemensions(std::vector<int64_t> Dims) {
	input_node_dims = Dims;
}

void OnnxInferenceBase::printTensorShape(const Ort::Value& tensor) {
    std::vector<int64_t> shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << " x ";
    }
    std::cout << std::endl;
}

// Helper function to get input or output names
void OnnxInferenceBase::getNodeNames(bool is_input) {
    Ort::AllocatorWithDefaultOptions allocator; // Create an allocator
    size_t node_count = is_input ? session.GetInputCount() : session.GetOutputCount();

    for (size_t i = 0; i < node_count; ++i) {
        if (is_input) {
          std::string input_name = session.GetInputNameAllocated(i, allocator).get();
          std::cout << input_name << std::endl;
        } else {
          std::string out_name = session.GetOutputNameAllocated(i, allocator).get();
          std::cout << out_name << std::endl;
        }
    }
}