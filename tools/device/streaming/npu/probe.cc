#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static double ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    std::string stage = "arguments";
    try {
        if (argc != 8 && argc != 10) {
            throw std::runtime_error("usage: probe cpu|qnn|proof MODEL INPUT D0,D1,D2,D3 THREADS REPEATS OUTPUT [PLUGIN BACKEND]");
        }
        const std::string mode = argv[1];
        const bool qnn = mode == "qnn" || mode == "proof";
        if ((mode != "cpu" && !qnn) || (qnn ? argc != 10 : argc != 8)) throw std::runtime_error("invalid mode or libraries");
        const int threads = std::stoi(argv[5]);
        const int repeats = std::stoi(argv[6]);
        if (threads < 1 || threads > 16 || repeats < 1 || repeats > 50) throw std::runtime_error("invalid thread or repeat count");
        std::vector<int64_t> shape;
        size_t count = 1;
        std::istringstream dims(argv[4]);
        for (std::string part; std::getline(dims, part, ',');) {
            const auto dim = std::stoll(part);
            if (dim < 1 || dim > 4096 || count * dim > 10000000) throw std::runtime_error("invalid dimension");
            shape.push_back(dim);
            count *= dim;
        }
        if (shape.size() != 4) throw std::runtime_error("expected NCHW input");
        stage = "input";
        std::vector<float> data(count);
        std::ifstream input(argv[3], std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(data.data()), count * sizeof(float)) || input.peek() != EOF) {
            throw std::runtime_error("input byte count does not match shape");
        }

        stage = "environment";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "murmur-npu-smoke");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        if (qnn) {
            stage = "register";
            env.RegisterExecutionProviderLibrary("QNNExecutionProvider", argv[8]);
            std::vector<Ort::ConstEpDevice> devices;
            for (const auto& d : env.GetEpDevices()) {
                if (std::string(d.EpName()) == "QNNExecutionProvider") devices.push_back(d);
            }
            if (devices.empty()) throw std::runtime_error("QNNExecutionProvider registered no devices");
            stage = "options";
            options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            std::unordered_map<std::string, std::string> ep{{"backend_path", argv[9]}, {"htp_performance_mode", "burst"}};
            if (mode == "proof") {
                ep.emplace("profiling_level", "basic");
                ep.emplace("profiling_file_path", "htp.csv");
                options.EnableProfiling("ort-proof");
            }
            options.AppendExecutionProvider_V2(env, devices, ep);
        }
        stage = "session";
        const auto init = ms();
        Ort::Session session(env, argv[2], options);
        const auto load = ms() - init;
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) throw std::runtime_error("expected one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        auto in = session.GetInputNameAllocated(0, allocator);
        auto out = session.GetOutputNameAllocated(0, allocator);
        const char* inputs[] = {in.get()};
        const char* outputs[] = {out.get()};
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, data.data(), count, shape.data(), shape.size());
        stage = "execute";
        auto run = [&] { return session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1); };
        const auto start = ms();
        auto result = run();
        const auto first = ms() - start;
        std::vector<double> times;
        for (int i = 0; i < repeats; ++i) {
            const auto t = ms();
            result = run();
            times.push_back(ms() - t);
        }
        std::sort(times.begin(), times.end());
        stage = "output";
        const auto info = result[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("expected float32 output");
        const auto* values = result[0].GetTensorData<float>();
        std::ofstream file(argv[7], std::ios::binary);
        file.write(reinterpret_cast<const char*>(values), info.GetElementCount() * sizeof(float));
        file.close();
        if (!file) throw std::runtime_error("cannot write output");
        if (mode == "proof") {
            auto path = session.EndProfilingAllocated(allocator);
            if (!*path.get() || std::rename(path.get(), "proof.json") != 0) throw std::runtime_error("cannot save ORT profile");
        }
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"mode\":\"" << mode << "\",\"ort\":\"" << Ort::GetVersionString()
                  << "\",\"strict\":" << (qnn ? "true" : "false") << ",\"threads\":" << threads
                  << ",\"repeats\":" << repeats << ",\"load_ms\":" << load << ",\"first_ms\":" << first
                  << ",\"warm_median_ms\":" << (times[(repeats - 1) / 2] + times[repeats / 2]) / 2
                  << ",\"warm_min_ms\":" << times.front() << ",\"out_shape\":[";
        const auto out_shape = info.GetShape();
        for (size_t i = 0; i < out_shape.size(); ++i) std::cout << (i ? "," : "") << out_shape[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "MURMUR_PROBE_FAILURE stage=" << stage << ": " << e.what() << '\n';
        return 10;
    }
}
