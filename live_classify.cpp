#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Función auxiliar para aplicar sigmoid por elemento
cv::Mat sigmoid(const cv::Mat& logits) {
    cv::Mat neg, expNeg;
    neg = -logits;
    cv::exp(neg, expNeg);
    return 1.0 / (1.0 + expNeg);
}

int main(int argc, char** argv) {
    // Procesar argumentos: -gpu o -cpu
    bool useGPU = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-gpu") == 0) {
            useGPU = true;
        } else if (std::strcmp(argv[i], "-cpu") == 0) {
            useGPU = false;
        }
    }

    // 1) Carga de etiquetas de ImageNet
    std::vector<std::string> classNames;
    {
        std::ifstream ifs("imagenet_classes.txt");
        if (!ifs.is_open()) {
            std::cerr << "Error: no se pudo abrir imagenet_classes.txt\n";
            return 1;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty())
                classNames.push_back(line);
        }
    }

    // 2) Carga del modelo ONNX
    cv::dnn::Net net = cv::dnn::readNetFromONNX("mobilenetv3-small-100.onnx");

    // 3) Configuración de backend/target según flag
    if (useGPU) {
        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Inferencia con CUDA activada.\n";
        } catch (...) {
            std::cerr << "Advertencia: no se pudo activar CUDA, usando CPU.\n";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "Inferencia forzada en CPU.\n";
    }

    // 4) Abrir stream MJPEG HTTP
    cv::VideoCapture cap("http://172.16.134.15:8080/video");
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el stream HTTP.\n";
        return 1;
    }

    // 5) Parámetros de preprocesamiento
    const cv::Size  inputSize(224,224);
    const double    scale     = 1.0/255.0;
    const cv::Scalar mean      = cv::Scalar(0.485,0.456,0.406);

    // Variables multilabel y FPS
    const float  multilabelThreshold = 0.3f;
    const int    maxLabels           = 2;
    auto lastInfer   = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    std::string currentLabel = "Esperando...";
    auto lastFPStime = std::chrono::steady_clock::now();
    int  frameCount  = 0;
    double fps       = 0.0;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frameCount++;

        auto now = std::chrono::steady_clock::now();
        // actualizar FPS cada segundo
        auto fpsElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFPStime).count();
        if (fpsElapsed >= 1000) {
            fps = frameCount * 1000.0 / fpsElapsed;
            frameCount = 0;
            lastFPStime = now;
        }

        // inferencia cada 2 segundos
        if (now - lastInfer >= std::chrono::seconds(2)) {
            cv::Mat blob = cv::dnn::blobFromImage(
                frame, scale, inputSize, mean, true, false
            );
            net.setInput(blob);
            cv::Mat logits = net.forward();       // 1×1000
            cv::Mat probs  = sigmoid(logits);     // sigmoid

            // recoger top-2 multilabel
            std::vector<std::pair<int,float>> picks;
            for (int i = 0; i < probs.cols; ++i) {
                float p = probs.at<float>(0,i);
                if (p > multilabelThreshold) picks.emplace_back(i,p);
            }
            std::sort(picks.begin(), picks.end(),
                      [](auto &a, auto &b){ return a.second > b.second; });
            if ((int)picks.size() > maxLabels) picks.resize(maxLabels);

            // construir etiqueta
            if (picks.empty()) {
                currentLabel = cv::format("Ninguna (>%.2f)", multilabelThreshold);
            } else {
                currentLabel.clear();
                for (size_t i = 0; i < picks.size(); ++i) {
                    int idx = picks[i].first;
                    float p = picks[i].second;
                    std::string name = idx < (int)classNames.size()
                                     ? classNames[idx]
                                     : cv::format("Clase%d", idx);
                    currentLabel += cv::format("%s(%.2f)", name.c_str(), p);
                    if (i+1 < picks.size()) currentLabel += ", ";
                }
            }
            lastInfer = now;
        }

        // dibujar multilabel
        int baseLine = 0;
        cv::Size ts = cv::getTextSize(currentLabel,
                                      cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::Point org(10, 10 + ts.height);
        cv::rectangle(frame,
                      org + cv::Point(0, baseLine),
                      org + cv::Point(ts.width, -ts.height),
                      cv::Scalar(0,0,0), cv::FILLED);
        cv::putText(frame, currentLabel, org,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 1);

        // dibujar FPS
        std::string fpsLabel = cv::format("FPS: %.1f", fps);
        int fpsBase = 0;
        cv::Size fpsTs = cv::getTextSize(fpsLabel,
                                         cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &fpsBase);
        cv::Point fpsOrg(frame.cols - fpsTs.width - 10, fpsTs.height + 10);
        cv::rectangle(frame,
                      fpsOrg + cv::Point(0, fpsBase),
                      fpsOrg + cv::Point(fpsTs.width, -fpsTs.height),
                      cv::Scalar(0,0,0), cv::FILLED);
        cv::putText(frame, fpsLabel, fpsOrg,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,0), 1);

        cv::imshow("Live Multilabel Classification", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
