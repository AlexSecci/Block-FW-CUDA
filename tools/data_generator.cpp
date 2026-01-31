#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <filesystem>
using namespace std;

void generate_graph(int n, int density_percent, const string& output_file) {
    float density = density_percent / 100.0f;
    cout << "Generating " << n << "x" << n << " adjacency matrix with density " << density_percent << "%..." << endl;

    vector<float> matrix(static_cast<size_t>(n) * n);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(0.0f, 100.0f);
    uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                matrix[static_cast<size_t>(i) * n + j] = 0.0f;
            } else {
                if (prob(gen) < density) {
                    matrix[static_cast<size_t>(i) * n + j] = dist(gen);
                } else {
                    matrix[static_cast<size_t>(i) * n + j] = numeric_limits<float>::infinity();
                }
            }
        }
    }

    if (!filesystem::exists("data")) {
        filesystem::create_directory("data");
    }
    ofstream outfile(output_file, ios::binary);
    if (!outfile) {
        cerr << "Error opening file: " << output_file << endl;
        return;
    }

    // Write N
    outfile.write(reinterpret_cast<const char*>(&n), sizeof(n));
    // Write matrix data
    outfile.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(float));

    outfile.close();
    cout << "Saved to " << output_file << endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <density_percent> <size1> [size2 ...]" << endl;
        cout << "Example: " << argv[0] << " 50 1024 2048 (50% density)" << endl;
        return 1;
    }

    int density_percent = stoi(argv[1]);
    if (density_percent < 0 || density_percent > 100) {
        cerr << "Error: Density must be between 0 and 100." << endl;
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        int n = stoi(argv[i]);
        if (n % 32 != 0) {
            cerr << "Error: Size " << n << " is not a multiple of 32." << endl;
            continue;
        }
        
        string filename = "data/graph_" + to_string(n) + "_" + to_string(density_percent) + ".dat";
        generate_graph(n, density_percent, filename);
    }

    return 0;
}
