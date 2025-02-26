#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <immintrin.h>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Parameters {
    struct ImageParams {
        int width = 256;
        int height = 256;
        int input_size = width * height;
    } image;
    
    struct PatternParams {
        double sine_frequency = 40.0;
        int checker_squares = 16;
    } pattern;
    
    struct NetworkParams {
        int latent_dim;
        double weight_init_range = 0.2;
    } network;
    
    struct TrainingParams {
        int epochs = 1000;
        double learning_rate = 0.001;
        int display_interval = 10;
    } training;
    
    struct OutputParams {
        std::string radial_file = "generated_radial.pgm";
        std::string sine_file = "generated_sine.pgm";
        std::string checker_file = "generated_checkerboard.pgm";
        std::string decoded_prefix = "final_decoded_";
        std::string decoded_suffix = ".pgm";
    } output;
    
    void init() {
        network.latent_dim = std::max(16, image.input_size / 200);
    }
};

struct Image {
    int width;
    int height;
    std::vector<double> data;
};

bool writePGM(const std::string& filename, const Image& img) {
    std::ofstream file(filename);
    if (!file) return false;
    file << "P2\n" << img.width << " " << img.height << "\n255\n";
    for (size_t i = 0; i < img.data.size(); i++) {
        int pixel = static_cast<int>(std::round(img.data[i] * 255));
        file << pixel << "\n";
    }
    return true;
}

Image generateRadialGradient(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);
    double centerX = width / 2.0;
    double centerY = height / 2.0;
    double maxDist = std::sqrt(centerX * centerX + centerY * centerY);
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            double dx = x - centerX;
            double dy = y - centerY;
            double dist = std::sqrt(dx * dx + dy * dy);
            double value = 1.0 - (dist / maxDist);
            img.data[y * width + x] = std::clamp(value, 0.0, 1.0);
        }
    }
    return img;
}

Image generateSineWaveImage(int width, int height) {
    double frequency = 40.0;
    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            double value = 0.5 * (1.0 + std::sin((static_cast<double>(x) / width) * frequency * 2 * M_PI) *
                                        std::sin((static_cast<double>(y) / height) * frequency * 2 * M_PI));
            img.data[y * width + x] = std::clamp(value, 0.0, 1.0);
        }
    }
    return img;
}

Image generateCheckerboard(int width, int height, int squares = 8) {
    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);
    int squareWidth = width / squares;
    int squareHeight = height / squares;
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            int cx = x / squareWidth;
            int cy = y / squareHeight;
            img.data[y * width + x] = ((cx + cy) % 2 == 0) ? 1.0 : 0.0;
        }
    }
    return img;
}

inline double dotProductSIMD(const double* a, const double* b, int n) {
    int i = 0;
    __m256d vsum = _mm256_setzero_pd();
    for (; i <= n - 4; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        vsum = _mm256_add_pd(vsum, _mm256_mul_pd(va, vb));
    }
    double buffer[4];
    _mm256_storeu_pd(buffer, vsum);
    double sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void vectorMultiplyAccumulateSIMD(double* dest, const double* a, double scalar, int n) {
    int i = 0;
    __m256d vscalar = _mm256_set1_pd(scalar);
    for (; i <= n - 4; i += 4) {
        __m256d vdest = _mm256_loadu_pd(dest + i);
        __m256d va = _mm256_loadu_pd(a + i);
        vdest = _mm256_add_pd(vdest, _mm256_mul_pd(vscalar, va));
        _mm256_storeu_pd(dest + i, vdest);
    }
    for (; i < n; i++) {
        dest[i] += scalar * a[i];
    }
}

inline void vectorSubtractSIMD(double* vec, const double* grad, double learning_rate, int n) {
    int i = 0;
    __m256d vlearning = _mm256_set1_pd(learning_rate);
    for (; i <= n - 4; i += 4) {
        __m256d vvec = _mm256_loadu_pd(vec + i);
        __m256d vgrad = _mm256_loadu_pd(grad + i);
        vvec = _mm256_sub_pd(vvec, _mm256_mul_pd(vgrad, vlearning));
        _mm256_storeu_pd(vec + i, vvec);
    }
    for (; i < n; i++) {
        vec[i] -= learning_rate * grad[i];
    }
}

inline void updateRowSIMD(double* weight, const double* grad, double update_factor, int n) {
    int i = 0;
    __m256d vupdate = _mm256_set1_pd(update_factor);
    for (; i <= n - 4; i += 4) {
        __m256d w = _mm256_loadu_pd(weight + i);
        __m256d g = _mm256_loadu_pd(grad + i);
        w = _mm256_sub_pd(w, _mm256_mul_pd(g, vupdate));
        _mm256_storeu_pd(weight + i, w);
    }
    for (; i < n; i++) {
        weight[i] -= update_factor * grad[i];
    }
}

inline double gatherDotProductSIMD(const double* W, const double* delta, int output_size, int latent_dim, int k) {
    int j = 0;
    __m256d vsum = _mm256_setzero_pd();
    for (; j <= output_size - 4; j += 4) {
        __m256i indices = _mm256_set_epi64x(((j+3)*latent_dim + k), ((j+2)*latent_dim + k), ((j+1)*latent_dim + k), (j*latent_dim + k));
        __m256d w_vec = _mm256_i64gather_pd(W, indices, 8);
        __m256d delta_vec = _mm256_loadu_pd(delta + j);
        vsum = _mm256_add_pd(vsum, _mm256_mul_pd(w_vec, delta_vec));
    }
    double buffer[4];
    _mm256_storeu_pd(buffer, vsum);
    double sum = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    for (; j < output_size; j++) {
        sum += W[j * latent_dim + k] * delta[j];
    }
    return sum;
}

class Decoder {
public:
    int output_size;
    int latent_dim;
    std::vector<double> W;
    std::vector<double> b;

    Decoder(int output_size, int latent_dim, double weight_init_range = 0.2)
        : output_size(output_size), latent_dim(latent_dim)
    {
        W.resize(output_size * latent_dim);
#pragma omp parallel for
        for (int i = 0; i < output_size; i++) {
            std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)) + i);
            std::uniform_real_distribution<double> dist(-0.5, 0.5);
            for (int j = 0; j < latent_dim; j++) {
                W[i * latent_dim + j] = dist(gen) * weight_init_range;
            }
        }
        b.resize(output_size, 0.0);
    }
    
    double sigmoid(double x) const { 
        return 1.0 / (1.0 + std::exp(-x)); 
    }
    
    std::vector<double> forward(const std::vector<double>& latent) const {
        std::vector<double> y(output_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            double sum = b[i];
            const double* w_row = &W[i * latent_dim];
            sum += dotProductSIMD(w_row, latent.data(), latent_dim);
            y[i] = sigmoid(sum);
        }
        return y;
    }
};

int main() {
    Parameters params;
    params.init();
    
    std::cout << "\033[1;36mGenerating synthetic images...\033[0m" << std::endl;
    Image imgRadial = generateRadialGradient(params.image.width, params.image.height);
    Image imgSine = generateSineWaveImage(params.image.width, params.image.height);
    Image imgChecker = generateCheckerboard(params.image.width, params.image.height, params.pattern.checker_squares);
    
    std::cout << "\033[1;32mSaving generated images...\033[0m" << std::endl;
    if (!writePGM(params.output.radial_file, imgRadial)) {
        std::cerr << "\033[1;31mError saving " << params.output.radial_file << "\033[0m" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "\033[1;32mSaved radial gradient to " << params.output.radial_file << "\033[0m" << std::endl;
    
    if (!writePGM(params.output.sine_file, imgSine)) {
        std::cerr << "\033[1;31mError saving " << params.output.sine_file << "\033[0m" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "\033[1;32mSaved sine wave to " << params.output.sine_file << "\033[0m" << std::endl;
    
    if (!writePGM(params.output.checker_file, imgChecker)) {
        std::cerr << "\033[1;31mError saving " << params.output.checker_file << "\033[0m" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "\033[1;32mSaved checkerboard to " << params.output.checker_file << "\033[0m" << std::endl;
    
    std::vector<Image> dataset = { imgRadial, imgSine, imgChecker };
    const int num_images = static_cast<int>(dataset.size());
    
    std::cout << "\033[1;36mInitializing latent codes...\033[0m" << std::endl;
    std::vector<std::vector<double>> latentCodes(num_images, std::vector<double>(params.network.latent_dim, 0.0));
#pragma omp parallel for
    for (int i = 0; i < num_images; i++) {
        std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr)) + i);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        for (int j = 0; j < params.network.latent_dim; j++) {
            latentCodes[i][j] = dist(gen) * params.network.weight_init_range;
        }
    }
    
    std::cout << "\033[1;36mInitializing decoder...\033[0m" << std::endl;
    Decoder decoder(params.image.input_size, params.network.latent_dim, params.network.weight_init_range);
    
    std::cout << "\033[1;36mStarting training on " << num_images << " images for " 
              << params.training.epochs << " iterations\033[0m" << std::endl;
    std::cout << "Output size: " << params.image.input_size 
              << ", Latent dimension: " << params.network.latent_dim << std::endl << std::endl;
    
    long long total_coeffs = static_cast<long long>(params.image.input_size) * params.network.latent_dim;
    double size_mb = (total_coeffs * sizeof(double)) / (1024.0 * 1024.0);
    
    std::cout << "Weight matrix W size: " 
          << params.image.input_size << " x " << params.network.latent_dim 
          << " = ";
    
    if (total_coeffs >= 1000000000) {
        std::cout << std::fixed << std::setprecision(2) << (total_coeffs / 1000000000.0) << " billion";
    } else if (total_coeffs >= 1000000) {
        std::cout << std::fixed << std::setprecision(2) << (total_coeffs / 1000000.0) << " million";
    } else if (total_coeffs >= 1000) {
        std::cout << std::fixed << std::setprecision(2) << (total_coeffs / 1000.0) << " thousand";
    } else {
        std::cout << total_coeffs;
    }
    
    std::cout << " coefficients (" 
          << std::fixed << std::setprecision(2) << size_mb
          << " MB)" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 1; epoch <= params.training.epochs; epoch++) {
        double total_loss = 0.0;
        std::vector<double> gradW(params.image.input_size * params.network.latent_dim, 0.0);
        std::vector<double> gradb(params.image.input_size, 0.0);
        
        for (int i = 0; i < num_images; i++) {
            std::vector<double> output = decoder.forward(latentCodes[i]);
            std::vector<double> delta(params.image.input_size, 0.0);
            double loss = 0.0;
            
            for (int j = 0; j < params.image.input_size; j++) {
                double diff = output[j] - dataset[i].data[j];
                loss += diff * diff;
                delta[j] = diff * output[j] * (1.0 - output[j]);
            }
            loss *= 0.5;
            total_loss += loss;
            
            for (int j = 0; j < params.image.input_size; j++) {
                double d = delta[j];
                vectorMultiplyAccumulateSIMD(&gradW[j * params.network.latent_dim],
                                           latentCodes[i].data(), d,
                                           params.network.latent_dim);
                gradb[j] += d;
            }
            
            std::vector<double> gradLatent(params.network.latent_dim, 0.0);
            for (int k = 0; k < params.network.latent_dim; k++) {
                gradLatent[k] = gatherDotProductSIMD(decoder.W.data(), delta.data(),
                                                   params.image.input_size, params.network.latent_dim, k);
            }
            vectorSubtractSIMD(latentCodes[i].data(), gradLatent.data(), params.training.learning_rate, params.network.latent_dim);
        }
        
        double update_factor = params.training.learning_rate / num_images;
        for (int j = 0; j < params.image.input_size; j++) {
            updateRowSIMD(&decoder.W[j * params.network.latent_dim],
                        &gradW[j * params.network.latent_dim],
                        update_factor,
                        params.network.latent_dim);
            decoder.b[j] -= update_factor * gradb[j];
        }
        
        if (epoch % params.training.display_interval == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            int hours = static_cast<int>(duration / 3600);
            int minutes = static_cast<int>((duration % 3600) / 60);
            int seconds = static_cast<int>(duration % 60);
            
            float progress = (epoch * 100.0f) / params.training.epochs;
            
            std::cout << "\r\033[K";
            
            std::cout << "\033[1;32m[";
            const int barWidth = 50;
            int pos = static_cast<int>(barWidth * progress / 100.0f);
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] ";
            
            std::cout << "\033[1;33m" << static_cast<int>(progress) << "% "
                      << "\033[1;35m - Average loss: " << std::fixed << std::setprecision(6) << (total_loss / num_images)
                      << "\033[1;34m - Time: " << hours << "h " << minutes << "m " << seconds << "s"
                      << "\033[0m" << std::endl;
            
            std::cout << "\033[1;36mLatent codes status:\033[0m" << std::endl;
            for (int i = 0; i < num_images; i++) {
                std::cout << "Image " << i << " latent norm: " 
                         << std::sqrt(std::inner_product(latentCodes[i].begin(), latentCodes[i].end(), 
                                                       latentCodes[i].begin(), 0.0)) << std::endl;
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    int total_hours = static_cast<int>(total_duration / 3600);
    int total_minutes = static_cast<int>((total_duration % 3600) / 60);
    int total_seconds = static_cast<int>(total_duration % 60);
    
    std::cout << std::endl << "\033[1;32mTraining completed in " << total_hours << "h " 
              << total_minutes << "m " << total_seconds << "s\033[0m" << std::endl;
    
    std::cout << "\033[1;36mReconstructing and saving images...\033[0m" << std::endl;
    for (int i = 0; i < num_images; i++) {
        std::vector<double> output = decoder.forward(latentCodes[i]);
        for (int j = 0; j < params.image.input_size; j++) {
            output[j] = std::clamp(output[j], 0.0, 1.0);
        }
        Image decoded;
        decoded.width = params.image.width;
        decoded.height = params.image.height;
        decoded.data = output;
        std::string filename = params.output.decoded_prefix + std::to_string(i) + params.output.decoded_suffix;
        
        double reconstruction_error = 0.0;
        for (int j = 0; j < params.image.input_size; j++) {
            double diff = output[j] - dataset[i].data[j];
            reconstruction_error += diff * diff;
        }
        reconstruction_error = std::sqrt(reconstruction_error / params.image.input_size);
        
        if (writePGM(filename, decoded)) {
            std::cout << "\033[1;32mImage " << i << " reconstructed and saved to '" << filename 
                      << "' (RMSE: " << reconstruction_error << ")\033[0m" << std::endl;
        } else {
            std::cerr << "\033[1;31mError saving image " << i << "\033[0m" << std::endl;
        }
    }
    
    return EXIT_SUCCESS;
}
