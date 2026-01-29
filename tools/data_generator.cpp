#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>

void generate_graph(int n, const std::string& output_file) {
    std::cout << "Generating " << n << "x" << n << " adjacency matrix..." << std::endl;

    std::vector<float> matrix(static_cast<size_t>(n) * n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                matrix[static_cast<size_t>(i) * n + j] = 0.0f;
            } else {
                matrix[static_cast<size_t>(i) * n + j] = dist(gen);
            }
        }
    }

    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file: " << output_file << std::endl;
        return;
    }

    // Write N
    outfile.write(reinterpret_cast<const char*>(&n), sizeof(n));
    // Write matrix data
    outfile.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(float));

    outfile.close();
    std::cout << "Saved to " << output_file << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <size1> [size2 ...]" << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        int n = std::stoi(argv[i]);
        if (n % 32 != 0) {
            std::cerr << "Error: Size " << n << " is not a multiple of 32." << std::endl;
            continue;
        }
        
        std::string filename = "data/graph_" + std::to_string(n) + ".dat";
        generate_graph(n, filename);
    }

    return 0;
}
