#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <filesystem>
using namespace std;

void generate_graph(int n, const string& output_file) {
    cout << "Generating " << n << "x" << n << " adjacency matrix..." << endl;

    vector<float> matrix(static_cast<size_t>(n) * n);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                matrix[static_cast<size_t>(i) * n + j] = 0.0f;
            } else {
                matrix[static_cast<size_t>(i) * n + j] = dist(gen);
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
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <size1> [size2 ...]" << endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        int n = stoi(argv[i]);
        if (n % 32 != 0) {
            cerr << "Error: Size " << n << " is not a multiple of 32." << endl;
            continue;
        }
        
        string filename = "data/graph_" + to_string(n) + ".dat";
        generate_graph(n, filename);
    }

    return 0;
}
