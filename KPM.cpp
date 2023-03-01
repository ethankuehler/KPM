#include <Eigen/Sparse>
#include <Eigen/Core>
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <omp.h>
#include <complex>


using namespace std::complex_literals;
typedef float real;
typedef Eigen::SparseMatrix<std::complex<real>, Eigen::RowMajor> mat;
typedef Eigen::Matrix<std::complex<real>, Eigen::Dynamic, Eigen::Dynamic> sub_mat;
typedef Eigen::VectorXcf vec;

void writeArrayToCSV(const std::vector<real> &arr, const std::string &filename) {
    std::ofstream file(filename);

    for (int i = 0; i < arr.size(); i++) {
        file << arr[i];

        if (i < arr.size() - 1) {
            file << ",";
        }
    }

    file.close();
}

std::vector<real> operator/(const std::vector<real> &v, real x) {
    std::vector<real> output = v;
    for (int i = 0; i < v.size(); i++) {
        output[i] = v[i] / x;
    }
    return output;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"

//Nx is the side length of the system.
void H_func(const vec &v, vec &o, const int L, real t, real tp, real m, float scale) {
    int x = 2;     //to next unit cell in x
    int y = 2*L;    //to next unit cell in y
    int z = (2*L)*(2*L); //to next unit cell in z
    int N = v.size();
    t = t/(2*scale);
    tp = tp/(2*scale);
    m = m/scale;
    for(int i = 0; i < N; i += 2){
        int a = i;
        int b = i + 1;
        //diagonal -m*sig_z term
        o[a] = -m*v[a];
        o[b] =  m*v[b];

        //+x upper half Tx term
        if (b + x < N and (i + 2) % (2 * L) != 0){
            o[a] +=  t*v[a + x] + tp*v[b + x];
            o[b] +=  tp*v[a + x] - t*v[b + x];
        }
        //+x lower half
        if (a - x >= 0 and (i + 2) % (2 * L) != 0) {
            o[a] +=  t*v[a - x] + tp*v[b - x];
            o[b] +=  tp*v[a - x] - t*v[b - x];
        }
        //+y upper half Ty term
        if (b + y < N and (i) % (2 * L * L) < 2 * (L - 1) * L) {
            o[a] += t * v[a + y] - 1if * tp * v[b + y];
            o[b] += 1if * tp * v[a + y] - t * v[b + y];
        }
        //+y lower half
        if (a - y >= 0 and (i) % (2 * L * L) < 2 * (L - 1) * L) {
            o[a] += t * v[a - y] - 1if * tp * v[b - y];
            o[b] += 1if * tp * v[a - y] - t * v[b - y];
        }
        //+z upper half
        if (b + z < N){
            o[a] +=  t*v[a + z];
            o[b] += -t*v[b + z];
        }
        //+z lower half
        if (a - z >= 0){
            o[a] +=  t*v[a - z];
            o[b] += -t*v[b - z];
        }
    }
}

std::vector<real> find_mu_low(const int order, int NR, real t, real tp, real m, int Nx, float scale) {
    std::vector<real> mu = std::vector<real>(order, 0);

    // create a random number generator
    std::random_device rd;  // obtain a random seed from the OS
    std::mt19937 eng(rd());  // seed the generator
    std::uniform_real_distribution<real> distr(0, 2 * M_PI);  // define the range of the distribution
    int size = (2*Nx)*(2*Nx)*(2*Nx);
#pragma omp parallel for
    for (int i = 0; i < NR; i++) {
        vec v(size);
        for (auto &j : v){
            j = std::exp(distr(eng) * 1if);
        }
        v.normalize();
        vec T0 = v;
        vec T1(size);
        H_func(v, T1, Nx, t, tp, m, scale);
        vec T2(size);
        std::complex<real> weight = 0;
        std::cout << "doing the " << i + 1 << " R" << std::endl;
        for (int j = 0; j < order; j++) {
            weight = v.adjoint()*T0;
            mu[j] += weight.real();
            H_func(T1, T2, Nx, t, tp, m, scale/2);
            //the scale is divied by two as its T2 = 2*H*T1 and scale is H/scale;
            T2 -= T0;
            T0 = T1;
            T1 = T2;
        }
    }

    return mu / float(NR);
}

#pragma clang diagnostic pop

int main(int argc, char **argv) {
    //define constants
    const real T = 1.0; //tight binding strength
    const real TP = 1.0;
    const real M = 1;
    const int NR = 100; //number of random vector samples
    const int order = 1000; // kpm order
    const int L = 50; // size of system

    omp_set_num_threads(12);

    std::cout << "starting solver" << std::endl;
    std::vector<real> mu = find_mu_low(order, NR, T, TP, M, L, 5);

    writeArrayToCSV(mu, "../mu.csv");
    return 0;
}