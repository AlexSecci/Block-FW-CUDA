#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "floyd_warshall_cpu.h"
#include "floyd_warshall_gpu.cuh"
#include "utils.h"
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

using namespace std;

int main(int argc, char** argv) {
    cout << "Floyd-Warshall CUDA Project Initialized" << endl;
    
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <mode> <graph input path> <output folder>" << endl;
        return 1;
    }
    cout << "Arguments parsed correctly" << endl;

    int mode = stoi(argv[1]);
    switch (mode)
    {
    case 0 :
        cout << "Mode selected: cpu-only" << endl;
        break;
    case 1 :
        cout << "Mode selected: gpu-only" << endl;
        break;
    case 2 :
        cout << "Mode selected: benchmark mode" << endl;
        break;
    default:
        cerr << "Error: Mode " << mode << " should be either 0 (cpu-only), 1 (gpu-only) or 2 (both + comparison)" << endl;
        return 1;
    }

    if (!filesystem::exists(argv[2])) {
            cerr << "Error: Input graph not found at path " << argv[2] << endl;
            return 1;
    }
    cout << "File " << argv[2] << " is present" << endl;

    ifstream infile(argv[2], ios::binary);
    if (!infile) {
        cerr << "Error opening file: " << argv[2] << endl;
        return 1;
    }
    cout << "File " << argv[2] << " opened correctly" << endl;

    int n;
    infile.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!infile) {
        cerr << "Error reading graph size from: " << argv[2] << endl;
        return 1;
    }
    cout << "Graph size: " << n << "x" << n << endl;

    vector<float> matrixCPU(static_cast<size_t>(n) * n);
    vector<float> matrixGPU(static_cast<size_t>(n) * n);
    if (mode == 0 || mode == 2) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float dist;
                infile.read(reinterpret_cast<char*>(&dist), sizeof(dist));
                if (!infile) {
                    cerr << "Error occurred while reading graph values" << endl;
                    return 1;
                }
                matrixCPU[static_cast<size_t>(i) * n + j] = dist;
            }
        }
        infile.close();
        if (mode == 2){
            matrixGPU = matrixCPU;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float dist;
                infile.read(reinterpret_cast<char*>(&dist), sizeof(dist));
                if (!infile) {
                    cerr << "Error occurred while reading graph values" << endl;
                    return 1;
                }
                matrixGPU[static_cast<size_t>(i) * n + j] = dist;
            }
        }
        infile.close();
    }
    
    cout << "Graph parsed correctly" << endl;
    writeGraphToFile(matrixCPU, string(argv[3]) + "\\InputCPU.txt");
    writeGraphToFile(matrixCPU, string(argv[3]) + "\\InputGPU.txt");
    if (mode == 0 ||  mode == 2) {
        // TODO: Run CPU version
        auto t1 = chrono::high_resolution_clock::now();
        floydWarshallCPU(matrixCPU, n);
        auto t2 = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> deltaTime = t2 - t1;
        cout << "Floyd-Warshall CPU execution completed in " << deltaTime.count() << " ms" << endl;
    }

    if (mode == 1 ||  mode == 2) {
        // TODO: Run GPU version
        auto t1 = chrono::high_resolution_clock::now();
        floydWarshallGPU(matrixGPU, n);
        auto t2 = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> deltaTime = t2 - t1;
        cout << "Floyd-Warshall GPU execution completed in " << deltaTime.count() << " ms" << endl;

    }

    if (mode == 2) {
        verifyResults(matrixCPU, matrixGPU);
    }

    writeGraphToFile(matrixCPU, string(argv[3]) + "\\OutputCPU.txt");
    writeGraphToFile(matrixGPU, string(argv[3]) + "\\OutputGPU.txt");
    // TODO: Output data

    return 0;
}
