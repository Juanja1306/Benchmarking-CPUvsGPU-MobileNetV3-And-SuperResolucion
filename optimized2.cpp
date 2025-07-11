#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_or_video_path>" << endl;
        return -1;
    }

    // Try opening as video
    VideoCapture cap(argv[1]);
    Mat frame, img_gray;
    bool isVideo = cap.isOpened();

    if (isVideo) {
        // Read first frame
        if (!cap.read(frame)) {
            cerr << "Cannot read video frame" << endl;
            return -1;
        }
        cvtColor(frame, img_gray, COLOR_BGR2GRAY);
    } else {
        // Load as image
        Mat img = imread(argv[1], IMREAD_GRAYSCALE);
        if (img.empty()) {
            cerr << "Cannot load image" << endl;
            return -1;
        }
        img_gray = img;
    }

    // Number of iterations to amplify workload
    const int ITER = 100;

    // Prepare CPU output
    Mat cpu_out;

    // Initialize GPU and stream
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "No CUDA device found." << endl;
        return -1;
    }
    cuda::setDevice(0);
    cuda::Stream stream;

    // Upload once to GPU
    cuda::GpuMat d_img, d_out;
    d_img.upload(img_gray, stream);
    stream.waitForCompletion();

    // Create Gaussian filter on GPU
    Ptr<cuda::Filter> gaussGPU = cuda::createGaussianFilter(
        CV_8UC1, CV_8UC1, Size(15,15), 3.0);

    // Warm-up GPU
    for (int i = 0; i < 10; ++i) {
        gaussGPU->apply(d_img, d_out, stream);
    }
    stream.waitForCompletion();

    // Time CPU workload
    double t0 = (double)getTickCount();
    for (int i = 0; i < ITER; ++i) {
        GaussianBlur(img_gray, cpu_out, Size(15,15), 3.0);
    }
    t0 = ((double)getTickCount() - t0) / getTickFrequency() * 1000.0;

    // Time GPU workload (processing only)
    double t1 = (double)getTickCount();
    for (int i = 0; i < ITER; ++i) {
        gaussGPU->apply(d_img, d_out, stream);
    }
    stream.waitForCompletion();
    t1 = ((double)getTickCount() - t1) / getTickFrequency() * 1000.0;

    cout << "CPU time for " << ITER << " blurs: " << t0 << " ms" << endl;
    cout << "GPU time for " << ITER << " blurs: " << t1 << " ms" << endl;
    cout << "Speedup: " << (t0 / t1) << "x" << endl;

    return 0;
}
