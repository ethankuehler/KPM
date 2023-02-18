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

void writeMatrixToCSV(const std::string &filename, const mat &matrix) {
    // Open the file stream
    std::ofstream file(filename);

    // Write the matrix to the file
    for (int i = 0; i < matrix.cols(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            auto t = matrix.coeff(i, j);
            file << t.real();
            if (j + 1 < matrix.cols()) {
                file << ',';
            }
        }
        file << "\n";
    }

    // Close the file stream
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
void H_func(const vec &v, vec &o, int L, real t, real tp, real m, float scale) {
    int x = 2;     //to next unit cell in x
    int y = 2*L;    //to next unit cell in y
    int z = (2*L)*(2*L); //to next unit cell in z
    int N = v.size();
    t = t/(2*scale);
    tp = tp/(2*scale);
    m = m/scale;
    //o.setZero();
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
            o[a] =  t*v[a + z];
            o[b] = -t*v[b + z];
        }
        //+z lower half
        if (a - z >= 0){
            o[a] =  t*v[a - z];
            o[b] = -t*v[b - z];
        }
    }
}

std::vector<real> find_mu(const mat &H, const int order, int NR) {
    std::vector<real> mu = std::vector<real>(order, 0);

    // create a random number generator
    std::random_device rd;  // obtain a random seed from the OS
    std::mt19937 eng(rd());  // seed the generator
    std::uniform_real_distribution<real> distr(0, 2 * M_PI);  // define the range of the distribution

#pragma omp parallel for
    for (int i = 0; i < NR; i++) {
        vec v(H.cols());
        for (auto &j : v){
            j = std::exp(distr(eng) * 1if);
        }
        v.normalize();
        vec T0 = v;
        vec T1 = H * v;
        vec T2;
        std::complex<real> weight = 0;
        std::cout << "doing the " << i + 1 << " R" << std::endl;
        for (int j = 0; j < order; j++) {
            weight = v.adjoint() * T0;
            mu[j] += weight.real();
            T2 = 2 * H * T1;
            T2 -= T0;
            T0 = T1;
            T1 = T2;
        }
    }

    return mu / NR;
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
            H_func(T1, T2, Nx, t, tp, m, scale);
            T2 = 2*T2;
            T2 -= T0;
            T0 = T1;
            T1 = T2;
        }
    }

    return mu / float(NR);
}

#pragma clang diagnostic pop

mat H2D(const int L, const real T) {
    int N = L * L;
    mat H = mat(N, N);

    for (int i = 0; i < N; i++) {
        if (i + 1 < N && ((i + 1) % L != 0)) {
            H.insert(i, i + 1) = -T;
        }
        if (i - 1 >= 0 && ((i) % L != 0)) {
            H.insert(i, i - 1) = -T;
        }
        if (i + L < N) {
            H.insert(i, i + L) = -T;
        }
        if (i - L >= 0) {
            H.insert(i, i - L) = -T;
        }
    }
    return H;
}

mat H1D(const int L, const real T) {
    mat H = mat(L, L);
    H.setZero();
    H.reserve(2 * L);
    for (int i = 0; i < L; i++) {
        if (i - 1 >= 0) {
            H.insert(i - 1, i) = -T;
        }
        if (i + 1 < L) {
            H.insert(i + 1, i) = -T;
        }
    }
    return H;
}

mat H3D(int L, real t) {
    int N = L * L * L;
    mat H = mat(N, N);

    for (int i = 1; i <= N; i++) {
        if (i - 1 > 0 && (i - 1) % L != 0) {
            H.insert(i - 1, i - 1 - 1) = t;
        }
        if (i + 1 <= N && i % L != 0) {
            H.insert(i - 1, i + 1 - 1) = t;
        }
        if (i + L <= N) {
            H.insert(i - 1, i + L - 1) = t;
        }
        if (i - L > 0) {
            H.insert(i - 1, i - L - 1) = t;
        }
        if (i + L * L <= N) {
            H.insert(i - 1, i + L * L - 1) = t;
        }
        if (i - L * L > 0) {
            H.insert(i - 1, i - L * L - 1) = t;
        }
    }
    return H;
}

//writes submatrix into matrix. i and j are top left corners.
void write_sub_marix(mat &tgt, const sub_mat &matrix, int i, int j) {
    int n = matrix.cols();
    for (int x = 0; x < n; x++) {
        for (int y = 0; y < n; y++) {
            tgt.insert(i + x, j + y) = matrix(x, y);
        }
    }
}

//hamiltonian for weyl model
//L is number of units cells and so number of atoms is (2*L)^3
mat HG(int L, real t, real tp, real m) {
    int N = (2 * L) * (2 * L) * (2 * L);
    mat H(N, N);

    sub_mat sx{{0, 1,},
               {1, 0}};
    sub_mat sy{{0,  -1if,},
               {1if, 0}};
    sub_mat sz{{1, 0},
               {0, -1}};

    sub_mat Tx = (t / 2) * sz + (tp / 2) * sx;
    sub_mat Ty = (t / 2) * sz + (tp / 2) * sy;
    sub_mat Tz = (t / 2) * sz;
    sub_mat diag = -m * sz;

    //2*3 sub matrices for forward and back and one diagonal.
    H.reserve(Eigen::VectorXi::Constant(N, 7 * 2));

    //we are going by two as we are iterating by unit cell
    for (int i = 0; i < N; i += 2) {
        if (i % (N / 100) == 0 || i % (N / 100) == 1) {
            std::cout << "H: " << (real(i) / real(N)) * 100 << std::endl;
        }
        write_sub_marix(H, diag, i, i); // writing diagonale4
        if ((i + 2 + 1 < N) and (i + 2) % (2 * L) != 0) {
            write_sub_marix(H, Tx, i, i + 2); // writing nu=x
            write_sub_marix(H, Tx, i + 2, i);
        }

        if ((i + (2 * L) + 1 < N) and (i) % (2 * L * L) < 2 * (L - 1) * L) {
            write_sub_marix(H, Ty, i, i + (2 * L)); // writing nu=y
            write_sub_marix(H, Ty, i + (2 * L), i);
        }
        if (i + (2 * L) * (2 * L) + 1 < N) {
            write_sub_marix(H, Ty, i, i + (2 * L) * (2 * L)); // writing nu=y
            write_sub_marix(H, Ty, i + (2 * L) * (2 * L), i);
        }
    }
    return H;
}

int main(int argc, char **argv) {
    //define constants
    const real T = 1.0; //tight binding strength
    const real TP = 1.0;
    const real M = 1.0;
    const int NR = 300; //number of random vector samples
    const int order = 1000; // kpm order
    const int L = 100; // size of system

    omp_set_num_threads(12);

    //mat H = HG(L, T, TP, M) / 4;
    std::cout << "H done\n";
    //H.makeCompressed();
    std::cout << "H compressed" << std::endl;

    //writeMatrixToCSV("../Cmatrix.csv", H);
    //std::cout << H << '\n';
    std::cout << "starting solver" << std::endl;
    std::vector<real> mu = find_mu_low(order, NR, T, TP, M, L, 4.0);
    //std::vector<real> mu = find_mu(H, order, NR);

    writeArrayToCSV(mu, "../mu.csv");
    /*
    vec v(H.cols());
    vec o1(H.cols());
    vec o2(H.cols());
    v.setRandom();

    o1 = H*v;
    H_func(v, o2, L, T, TP, M, 4.0);

    bool same = true;
    for (int i = 0; i < o1.size(); i++){
        if (o1[i] != o2[i]) {
            same = false;
        } else {
            std::cout << "same at " << i << '\n';
        }
    }
    std::cout << "they are " << same << '\n';
     */
    return 0;
}