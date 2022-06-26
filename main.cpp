#include <iostream>
#include <vector>
//Thread building blocks library
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_reduce.h>
//Free Image library
#include <FreeImagePlus.h>
#include <math.h>
//Chrono Library
#include <chrono>
//Random Library
#include <random>

using namespace std;
using namespace tbb;

vector<float> calculateGaussianMatrix(float sigma, int kernel_size)
{
    vector<float> kernel;

    int x_kernel = kernel_size / 2;
    int y_kernel = kernel_size / 2;

    // sequential version
    // calculating gaussian matrix
    for (int y = -y_kernel;  y <= y_kernel; y++)
    {
        for (int x = -x_kernel; x <= x_kernel; x++)
        {
            kernel.push_back(((1 / (2 * M_PI * pow(sigma, 2))) * exp(- ((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))))));
        }
    }

    float total = 0.0f;

    // normalising gaussian matrix
    for (int i = 0; i <= kernel.size() - 1; i++)
    {
        total += kernel[i];
    }

    for (int i = 0; i <= kernel.size() - 1; i++)
    {
        kernel[i] = kernel[i] / total;
    }

    return kernel;
}

void sequentialGaussian(vector<float> kernel, int kernel_size)
{
    if (kernel_size % 2 == 0)
    {
        cout << "Kernel size must be odd!\n" << "Exiting..." << endl;

        return;
    }

    // setup and load the input image array
    fipImage inputImageGaussian;
    inputImageGaussian.load("../Images/render_1.png");
    //inputImageGaussian.load("../Images/Barcelona_lowres.jpg");
    inputImageGaussian.convertToFloat();

    // obtain width and height of image
    auto width_gauss = inputImageGaussian.getWidth();
    auto height_gauss = inputImageGaussian.getHeight();

    const float* const inputBuffer = (float*)inputImageGaussian.accessPixels();

    // setup output image array
    fipImage outputImageGaussian;
    outputImageGaussian = fipImage(FIT_FLOAT, width_gauss, height_gauss, 32);

    float *outputBuffer = (float*)outputImageGaussian.accessPixels();

    // get total array size
    uint64_t numElements = width_gauss * height_gauss;

    int kernel_width_offset = kernel_size / 2;
    int kernel_height_offset = kernel_size / 2;

    // take current time before task
    auto sequential_time_start = chrono::high_resolution_clock::now();

    // image coordinates
    for (int y_image = 0; y_image < height_gauss; y_image++)
    {
        for (int x_image = 0; x_image < width_gauss; x_image++)
        {
            float sum = 0.0f;

            // kernel coordinates
            for (int y_kernel = -kernel_height_offset; y_kernel != kernel_height_offset; y_kernel++)
            {
                for (int x_kernel = -kernel_width_offset; x_kernel != kernel_width_offset; x_kernel++)
                {
                    // current position of the kernel and image
                    int kernel_pos = (y_kernel + kernel_height_offset) * kernel_size + (x_kernel + kernel_width_offset);
                    int current_pos = ((y_image + y_kernel) * width_gauss) + (x_image + x_kernel);

                    if (current_pos >= 0 && current_pos <= numElements)
                    {
                        sum += kernel[kernel_pos] * inputBuffer[current_pos];
                    }
                }
            }

            outputBuffer[y_image * width_gauss + x_image] = sum;
        }
    }

    // take current time after task
    auto sequential_time_end = chrono::high_resolution_clock::now();

    // convert time difference and output duration
    auto sequential_duration = chrono::duration_cast<chrono::microseconds>(sequential_time_end - sequential_time_start);
    cout << "Time taken (Sequential): " << (sequential_duration.count() / 1e+6) << " microseconds." << endl;

    //Save the processed image
    outputImageGaussian.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    outputImageGaussian.convertTo24Bits();
    outputImageGaussian.save("sequentialGauss.png");
}

void parallelGaussian(vector<float> kernel, int kernel_size, int stepSize)
{
    if (kernel_size % 2 == 0)
    {
        cout << "Kernel size must be odd!\n" << "Exiting..." << endl;

        return;
    }

    // setup and load the input image array
    fipImage inputImageGaussian;
    //inputImageGaussian.load("../Images/render_1.png");
    inputImageGaussian.load("../Images/Barcelona_lowres.jpg");
    inputImageGaussian.convertToFloat();

    // obtain width and height of image
    auto width_gauss = inputImageGaussian.getWidth();
    auto height_gauss = inputImageGaussian.getHeight();

    const float* const inputBuffer = (float*)inputImageGaussian.accessPixels();

    // setup output image array
    fipImage outputImageGaussian;
    outputImageGaussian = fipImage(FIT_FLOAT, width_gauss, height_gauss, 32);

    float *outputBuffer = (float*)outputImageGaussian.accessPixels();

    // get total array size
    uint64_t numElements = width_gauss * height_gauss;

    int kernel_width_offset = kernel_size / 2;
    int kernel_height_offset = kernel_size / 2;

    // take current time before task
    auto parallel_time_start = chrono::high_resolution_clock::now();

    // blocked range2D
    // [=] = capture all by reference
    parallel_for(blocked_range2d<int, int>(0, height_gauss, stepSize, 0, width_gauss, stepSize), [=](const blocked_range2d<int, int>& range) {

        // image coordinates
        for (int y_image = range.rows().begin(); y_image < range.rows().end(); y_image++)
        {
            for (int x_image = range.cols().begin(); x_image < range.cols().end(); x_image++)
            {
                float sum = 0.0f;

                // kernel coordinates
                for (int y_kernel = -kernel_height_offset; y_kernel != kernel_height_offset; y_kernel++)
                {
                    for (int x_kernel = -kernel_width_offset; x_kernel != kernel_width_offset; x_kernel++)
                    {
                        // current position of the kernel and image
                        int kernel_pos = (y_kernel + kernel_height_offset) * kernel_size + (x_kernel + kernel_width_offset);
                        int current_pos = ((y_image + y_kernel) * width_gauss) + (x_image + x_kernel);

                        if (current_pos >= 0 && current_pos <= numElements)
                        {
                            sum += kernel[kernel_pos] * inputBuffer[current_pos];
                        }
                    }
                }

                outputBuffer[y_image * width_gauss + x_image] = sum;
            }
        }
    });

    // take current time after task
    auto parallel_time_end = chrono::high_resolution_clock::now();

    // convert time difference and output duration
    auto parallel_duration = chrono::duration_cast<chrono::microseconds>(parallel_time_end - parallel_time_start);
    cout << "Time taken (Parallel): " << (parallel_duration.count() / 1e+6) << " microseconds." << endl;

    //Save the processed image
    //outputImageGaussian.convertToType(FREE_IMAGE_TYPE::FIT_BITMAP);
    //outputImageGaussian.convertTo24Bits();
    //outputImageGaussian.save("gaussianParallel.png");
}

void sequentialRGB(int threshold)
{
    // Setup Input image array
    fipImage inputImage;
    inputImage.load("../Images/render_1.png");

    fipImage inputImage2;
    inputImage2.load("../Images/render_2.png");

    // only need width/height from one image as both
    // images have same values (5000 x 7000)
    unsigned int width = inputImage.getWidth();
    unsigned int height = inputImage.getHeight();

    // Setup Output image array
    fipImage outputImage;
    outputImage = fipImage(FIT_BITMAP, width, height, 24);

    //2D Vector to hold the RGB colour data of an image
    vector<vector<RGBQUAD>> rgbValues;
    rgbValues.resize(height, vector<RGBQUAD>(width));

    vector<vector<RGBQUAD>> rgbValues2;
    rgbValues2.resize(height, vector<RGBQUAD>(width));

    vector<vector<RGBQUAD>> outputrgbValues;
    outputrgbValues.resize(height, vector<RGBQUAD>(width));

    RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel
    RGBQUAD rgb2;



    //Extract colour data from image and store it as individual RGBQUAD elements for every pixel
    for(int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            inputImage.getPixelColor(x, y, &rgb); //Extract pixel(x,y) colour data and place it in rgb
            inputImage2.getPixelColor(x, y, &rgb2); //Extract pixel(x,y) colour data and place it in rgb

            rgbValues[y][x].rgbRed = rgb.rgbRed;
            rgbValues[y][x].rgbGreen = rgb.rgbGreen;
            rgbValues[y][x].rgbBlue = rgb.rgbBlue;

            rgbValues2[y][x].rgbRed = rgb2.rgbRed;
            rgbValues2[y][x].rgbGreen = rgb2.rgbGreen;
            rgbValues2[y][x].rgbBlue = rgb2.rgbBlue;

            // if image difference is greater than the threshold (1), set pixel to white
            // else leave pixel black
            // 'threshold' limit used to define image intensity
            if ((abs(rgbValues[y][x].rgbRed - rgbValues2[y][x].rgbRed) > threshold) &&
                (abs(rgbValues[y][x].rgbGreen - rgbValues2[y][x].rgbGreen) > threshold) &&
                (abs(rgbValues[y][x].rgbBlue - rgbValues2[y][x].rgbBlue) > threshold))
            {
                outputrgbValues[y][x].rgbRed = 255;
                outputrgbValues[y][x].rgbGreen = 255;
                outputrgbValues[y][x].rgbBlue = 255;
            }
        }
    }

    //Place the pixel colour values into output image
    for(int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            outputImage.setPixelColor(x, y, &outputrgbValues[y][x]);
        }
    }

    //Save the processed image
    outputImage.save("RGB_processed.png");

    // count the number of white pixels in the threshold image output
    // and calculate the percentage
    int numWhitePixels = 0;

    for(int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (outputrgbValues[y][x].rgbRed == 255 &&
                outputrgbValues[y][x].rgbGreen == 255 &&
                outputrgbValues[y][x].rgbBlue == 255)
            {
                numWhitePixels++;
            }
        }
    }

    cout << "Number of white pixels: " << numWhitePixels << endl;
    cout << "Percentage of white pixels: " << ((float)numWhitePixels / (width * height)) * 100 << "%" << endl;

    // place red pixel in random location and search for it
    random_device rd;   // obtain random number from hardware
    mt19937 mt(rd());   // seed the generator
    // defining range for both x and y
    uniform_int_distribution<> y_dist(0, height);
    uniform_int_distribution<> x_dist(0, width);

    // setting generated index for x and y
    int y_rand = y_dist(mt);
    int x_rand = x_dist(mt);

    outputrgbValues[y_rand][x_rand].rgbRed = 255;
    outputrgbValues[y_rand][x_rand].rgbGreen = 0;
    outputrgbValues[y_rand][x_rand].rgbBlue = 0;

    outputImage.setPixelColor(x_rand, y_rand, &outputrgbValues[y_rand][x_rand]);

    cout << "Red pixel set at X_rand: " << x_rand << ", Y_rand: " << y_rand << endl;

    // randomly place red pixel in image
    for(int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (outputrgbValues[y][x].rgbRed == 255 &&
                outputrgbValues[y][x].rgbGreen == 0 &&
                outputrgbValues[y][x].rgbBlue == 0)
            {
                cout << "Red pixel found! X: " << x << ", Y: " << y << endl;
            }
        }
    }
}

void parallelRGB(int threshold, int stepSize)
{
    // Setup Input image array
    fipImage inputImage;
    inputImage.load("../Images/render_1.png");

    fipImage inputImage2;
    inputImage2.load("../Images/render_2.png");

    // only need width/height from one image as both
    // images have same values (5000 x 7000)
    unsigned int width = inputImage.getWidth();
    unsigned int height = inputImage.getHeight();

    // Setup Output image array
    fipImage outputImage;
    outputImage = fipImage(FIT_BITMAP, width, height, 24);

    //2D Vector to hold the RGB colour data of an image
    vector<vector<RGBQUAD>> rgbValues;
    rgbValues.resize(height, vector<RGBQUAD>(width));

    vector<vector<RGBQUAD>> rgbValues2;
    rgbValues2.resize(height, vector<RGBQUAD>(width));

    vector<vector<RGBQUAD>> outputrgbValues;
    outputrgbValues.resize(height, vector<RGBQUAD>(width));

    //Extract colour data from image and store it as individual RGBQUAD elements for every pixel
    parallel_for(blocked_range2d<int, int>(0, height, stepSize, 0, width, stepSize), [&](const blocked_range2d<int, int>& range) {

        // declared within the parallel_for as its being accessed
        // by reference by multiple threads
        RGBQUAD rgb;  //FreeImage structure to hold RGB values of a single pixel
        RGBQUAD rgb2;

        for (int y = range.rows().begin(); y != range.rows().end(); y++)
        {
            for (int x = range.cols().begin(); x != range.cols().end(); x++)
            {
                inputImage.getPixelColor(x, y, &rgb); //Extract pixel(x,y) colour data and place it in rgb
                inputImage2.getPixelColor(x, y, &rgb2); //Extract pixel(x,y) colour data and place it in rgb

                rgbValues[y][x].rgbRed = rgb.rgbRed;
                rgbValues[y][x].rgbGreen = rgb.rgbGreen;
                rgbValues[y][x].rgbBlue = rgb.rgbBlue;

                rgbValues2[y][x].rgbRed = rgb2.rgbRed;
                rgbValues2[y][x].rgbGreen = rgb2.rgbGreen;
                rgbValues2[y][x].rgbBlue = rgb2.rgbBlue;

                // if pixel value difference is greater than the threshold (1), set pixel to white
                // else leave pixel black
                // threshold limit used to define image intensity
                if ((abs(rgbValues[y][x].rgbRed - rgbValues2[y][x].rgbRed) > threshold) &&
                    (abs(rgbValues[y][x].rgbGreen - rgbValues2[y][x].rgbGreen) > threshold) &&
                    (abs(rgbValues[y][x].rgbBlue - rgbValues2[y][x].rgbBlue) > threshold))
                {
                    outputrgbValues[y][x].rgbRed = 255;
                    outputrgbValues[y][x].rgbGreen = 255;
                    outputrgbValues[y][x].rgbBlue = 255;
                }
            }
        }
    });

    // count the number of white pixels in the threshold image output
    // and calculate the percentage
    // parallel_reduction:
    // final result is returned in 'numWhitePixels'
    // '0' - identity value - for addition this is 0
    int numWhitePixels = parallel_reduce(blocked_range2d<int, int>(0, height, stepSize, 0, width, stepSize), 0,

            // process sub_range setup by TBB
            // the lambda takes a range and an initValue parameter
            // which has the value of the 'identity' parameter (0)
            // provided above
                                         [=](const blocked_range2d<int>& range, int initValue)-> int
                                         {
                                             for (int y = range.rows().begin(); y != range.rows().end(); y++)
                                             {
                                                 for (int x = range.cols().begin(); x != range.cols().end(); x++)
                                                 {
                                                     if (outputrgbValues[y][x].rgbRed == 255 &&
                                                         outputrgbValues[y][x].rgbGreen == 255 &&
                                                         outputrgbValues[y][x].rgbBlue == 255)
                                                     {
                                                         initValue++;
                                                     }
                                                 }
                                             }

                                             return initValue;
                                         },

            // TBB calls this to combine the results from 2 sub-ranges (x & y)
            // which forms part of the final result
                                         [&](int x, int y)-> int
                                         {
                                             return x + y;
                                         }
    );

    cout << "Number of white pixels: " << numWhitePixels << endl;
    // numWhitePixels cast to float to calculate percentage of total white pixels in image
    cout << "Percentage of white pixels: " << ((float)numWhitePixels / (width * height)) * 100 << "%" << endl;

    //Place the pixel colour values into output image
    parallel_for(blocked_range2d<int, int>(0, height, stepSize, 0, width, stepSize), [&](blocked_range2d<int, int>& range) {

        for (int y = range.rows().begin(); y != range.rows().end(); y++)
        {
            for (int x = range.cols().begin(); x != range.cols().end(); x++)
            {
                outputImage.setPixelColor(x, y, &outputrgbValues[y][x]);
            }
        }
    });

    //Save the processed image
    outputImage.save("RGB_processed.png");

    random_device rd;   // obtain random number from hardware
    mt19937 mt(rd());   // seed the generator
    // defining range for both x and y
    uniform_int_distribution<> y_dist(0, height);
    uniform_int_distribution<> x_dist(0, width);

    // setting generated index for x and y
    int y_rand = y_dist(mt);
    int x_rand = x_dist(mt);

    outputrgbValues[y_rand][x_rand].rgbRed = 255;
    outputrgbValues[y_rand][x_rand].rgbGreen = 0;
    outputrgbValues[y_rand][x_rand].rgbBlue = 0;

    // placing red pixel in random location within the image
    outputImage.setPixelColor(x_rand, y_rand, &outputrgbValues[y_rand][x_rand]);

    cout << "Red pixel set at X_rand: " << x_rand << ", Y_rand: " << y_rand << endl;

    int x_returnIndex = -1;   // store index where the red pixel is located
    int y_returnIndex = -1;

    // cancellation
    parallel_for(blocked_range2d<int, int>(0, height, stepSize, 0, width, stepSize), [&](blocked_range2d<int, int>& range) {

        for (int y = range.rows().begin(); y != range.rows().end(); y++)
        {
            for (int x = range.cols().begin(); x != range.cols().end(); x++)
            {
                if (outputrgbValues[y][x].rgbRed == 255 &&
                    outputrgbValues[y][x].rgbGreen == 0 &&
                    outputrgbValues[y][x].rgbBlue == 0)
                {
                    // when the red pixel is found, tell the task group
                    // to cancel and store the index the red pixel is
                    // located at in x and y return index
                    if (task::self().cancel_group_execution())
                    {
                        // the method returns true if it actually causes cancellation
                        // false if the task_group_context was already cancelled
                        x_returnIndex = x;
                        y_returnIndex = y;

                        cout << "Cancelling at index: X: " << x << " Y: " << y << endl;
                    }
                }
            }
        }
    });

    cout << "Cancelling at index: X: " << x_returnIndex << " Y: " << y_returnIndex << endl;
}

int main()
{
    // change number of threads for testing in report
    // default number of threads = 2
    int nt = task_scheduler_init::default_num_threads();
    task_scheduler_init T(nt);

    //Part 1 (Greyscale Gaussian blur): -----------DO NOT REMOVE THIS COMMENT----------------------------//


    // gaussian kernel user-defined values
/*    int kernel_size = 14;
    float sigma = 20.0f;

    vector<float> kernel; // gaussian kernel array
    kernel = calculateGaussianMatrix(sigma, kernel_size);   // calculating kernel matrix for use
    sequentialGaussian(kernel, kernel_size);    // creating gaussian image sequentially
    parallelGaussian(kernel, kernel_size, 5);   // creating gaussian image parallel*/

    // code for testing purposes in report
/*  int stepSize = 10;
    for (int i = 1; i <= 10; i++)
    {
        cout << "Test: " << i << endl;
        parallelGaussian(kernel, kernel_size, stepSize);
        cout << endl;
    }*/

    //Part 2 (Colour image processing): -----------DO NOT REMOVE THIS COMMENT----------------------------//

    //sequentialRGB(1);
    parallelRGB(1, 1);

    return 0;
}