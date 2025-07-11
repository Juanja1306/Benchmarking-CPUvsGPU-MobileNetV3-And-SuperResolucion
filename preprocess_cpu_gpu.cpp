#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;
using namespace cv;

// Helper to measure time per invocation
class Timer {
    TickMeter tm;
public:
    void start() { tm.reset(); tm.start(); }
    void stop()  { tm.stop();  }
    double get() const { return tm.getTimeMilli(); }
};

int main(int argc, char* argv[])
{
    // Allow default webcam if no argument given
    string input = (argc >= 2 ? argv[1] : "");
    VideoCapture cap;
    Mat single;
    bool isVideo = false;
    if (input.empty()) {
        cap.open(0);
        isVideo = cap.isOpened();
    } else {
        cap.open(input);
        isVideo = cap.isOpened();
    }
    if (!isVideo) {
        single = imread(input, IMREAD_GRAYSCALE);
        if (single.empty()) { cerr << "Cannot open input" << endl; return -1; }
    }

    // Initialize GPU
    int gpuCount = cuda::getCudaEnabledDeviceCount();
    if (gpuCount == 0) { cerr << "No CUDA device found." << endl; return -1; }
    cuda::setDevice(0);

    // Prepare GPU filters
    Ptr<cuda::Filter> gaussianGPU = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5,5), 1.5);
    Ptr<cuda::CannyEdgeDetector> cannyGPU = cuda::createCannyEdgeDetector(50.0, 150.0);
    Ptr<cuda::Filter> erodeGPU   = cuda::createMorphologyFilter(MORPH_ERODE,  CV_8UC1,
                                  getStructuringElement(MORPH_RECT, Size(3,3)));
    Ptr<cuda::Filter> dilateGPU  = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1,
                                  getStructuringElement(MORPH_RECT, Size(3,3)));

    Timer tCPU, tGPU;
    int frames = 0;
    double totalTimeCPU = 0, totalTimeGPU = 0;

    while (true) {
        Mat frame, gray;
        if (isVideo) {
            if (!cap.read(frame)) break;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        } else {
            gray = single;
        }

        // CPU pipeline timing
        tCPU.start();
        Mat cpu_gauss, cpu_erode, cpu_dilate, cpu_edges, cpu_eq;
        GaussianBlur(gray, cpu_gauss, Size(5,5), 1.5);
        erode(cpu_gauss, cpu_erode, getStructuringElement(MORPH_RECT, Size(3,3)));
        dilate(cpu_erode, cpu_dilate, getStructuringElement(MORPH_RECT, Size(3,3)));
        Canny(cpu_dilate, cpu_edges, 50, 150);
        equalizeHist(gray, cpu_eq);
        tCPU.stop();

        // GPU-only pipeline timing
        tGPU.start();
        cuda::GpuMat d_gray(gray), d_gauss, d_erode, d_dilate, d_edges, d_eq;
        gaussianGPU->apply(d_gray, d_gauss);
        erodeGPU->apply(d_gauss, d_erode);
        dilateGPU->apply(d_erode, d_dilate);
        cannyGPU->detect(d_dilate, d_edges);
        cuda::equalizeHist(d_gray, d_eq);
        // synchronize via download operations (download() blocks until ready)
        tGPU.stop();

        totalTimeCPU += tCPU.get();
        totalTimeGPU += tGPU.get();
        frames++;

        // Download GPU results
        Mat gpu_edges, gpu_eq;
        d_edges.download(gpu_edges);
        d_eq.download(gpu_eq);

        // Display results
        Mat top;
        hconcat(cpu_edges, gpu_edges, top);
        imshow("Edges: CPU | GPU", top);

        Mat bottom;
        hconcat(cpu_eq, gpu_eq, bottom);
        imshow("Hist EQ: CPU | GPU", bottom);

        if (waitKey(1) == 27) break;
        if (!isVideo) break;
    }

    cout << "Processed frames: " << frames << endl;
    cout << "Avg CPU time/frame: " << (totalTimeCPU/frames) << " ms" << endl;
    cout << "Avg GPU time/frame: " << (totalTimeGPU/frames) << " ms" << endl;

    return 0;
}
