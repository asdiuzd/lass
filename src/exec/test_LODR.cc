#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>

#include "json.h"

using namespace std;
using namespace cv;
using namespace ceres;
using json = nlohmann::json;

namespace lass {

#define EMBEDDING_LENGTH 12
#define StandardCosTheta 0.5
// cos(PI / 3)

// local equilibrium
class LEAutoDiffCost {
public:
    LEAutoDiffCost(double w): w_(w) {}
    template<typename T>
    bool operator() (
        const T* const yi,
        const T* const yj,
        const T* const yl,
        T* e
    ) const {
        e[0] = T(0);
        auto li = T(0), lj = T(0);
        // e[0] = w_ * (yi[0] - yl[0]) * (yj[0] - yl[0]);

        for(int idx = 0; idx < EMBEDDING_LENGTH; idx++) {
            e[0] += w_ * (yi[idx] - yl[idx]) * (yj[idx] - yl[idx]);
            li += yi[idx] * yi[idx];
            lj += yj[idx] * yj[idx];
        }

        e[0] /= ceres::sqrt(li) * ceres::sqrt(lj);
        e[0] -= StandardCosTheta;

        return true;
    }

private:
    const double w_;
};

// local orthogonality
class LOAutoDiffCost {
public:
    LOAutoDiffCost(double w): w_(w) {}

    template <typename T>
    bool operator()(
        const T* const yi,
        const T* const yj,
        const T* const yl,
        T* e
    ) const {
        e[0] = w_ * (yi[0] - yl[0]) * (yj[0] - yl[0]);

        for(int idx = 1; idx < EMBEDDING_LENGTH; idx++) {
            e[0] += w_ * (yi[idx] - yl[idx]) * (yj[idx] - yl[idx]);
        }
        return true;
    }
private:
    const double w_;
};

class DistanceAutoDiffCost {
public:
    DistanceAutoDiffCost(double w, double d): w_(w), d_(d){}

    template <typename T>
    bool operator()(
        const T* const yi,
        const T* const yj,
        T* e
    ) const {
        auto dis = (yi[0] - yj[0]) * (yi[0] - yj[0]);
        for(int idx = 1; idx < EMBEDDING_LENGTH; idx++) {
            dis += (yi[idx] - yj[idx]) * (yi[idx] - yj[idx]);
        }
        e[0] = w_* (ceres::sqrt(dis) - d_);

        return true;
    }

private:
    const double w_;
    const double d_;
};

void import_coding_book(const char *fn, vector<vector<double>>& coding_book) {
    LOG(INFO) << "import coding book..." << endl;
    json js;
    ifstream ifn(fn);
    ifn >> js;

    const int node_number = js.size();
    coding_book.resize(node_number, vector<double>(EMBEDDING_LENGTH, 0));
    LOG(INFO) << "node number = " << node_number << endl;
    LOG(INFO) << "embedding length = " << EMBEDDING_LENGTH << endl;

    for (int i = 0; i < node_number; i++) {
        CHECK(js[i].size() == EMBEDDING_LENGTH) << "Inconsist embedding length: " << js[i].size() << " - " << EMBEDDING_LENGTH << endl;

        for (int j = 0; j < EMBEDDING_LENGTH; j++) {
            coding_book[i][j] = js[i][j].get<double>();
        }
    }

    LOG(INFO) << "success." << endl;
}

void export_coding_book(const char *fn, vector<vector<double>>& coding_book) {
    LOG(INFO) << "export coding book..." << endl;
    json js(coding_book);
    ofstream ofn(fn);

    ofn << js;
}

void import_distance_matrix(const char *fn, vector<vector<double>>& distance_matrix) {
    LOG(INFO) << "import distance matrix..." << endl;
    json js;
    ifstream ifn(fn);
    ifn >> js;

    const int node_number = js.size();
    distance_matrix.resize(node_number, vector<double>(node_number, 0));
    LOG(INFO) << "node number = " << node_number << endl;

    for (int i = 0; i < node_number; i++) {
        CHECK(js[i].size() == node_number) << "Inconsistency matrix!" << endl;

        for (int j = 0; j < node_number; j++) {
            distance_matrix[i][j] = js[i][j].get<double>();
        }
    }

    LOG(INFO) << "success." << endl;
}

double compute_LO_loss(vector<vector<double>>& coding_book, vector<vector<int>>& adjacency_table) {
    double loss = 0;
    for (int idx = 0; idx < coding_book.size(); idx++) {
        auto& yl = coding_book[idx];
        auto& adj = adjacency_table[idx];

        for (int idx_i = 0; idx_i < adj.size(); idx_i++) {
            auto& yi = coding_book[adj[idx_i]];

            for (int idx_j = 0; idx_j < idx_i; idx_j++) {
                auto& yj = coding_book[adj[idx_j]];

                double res = 0;
                for (int idxx = 0; idxx < yi.size(); idxx++) {
                    res += (yi[idxx] - yl[idxx]) * (yj[idxx] - yl[idxx]);
                }
                loss += res * res;
            }
        }
    }

    return loss;
}

double compute_distance_loss(vector<vector<double>>& coding_book, vector<vector<double>>& distance_matrix) {
    double loss = 0;
    for (int i = 0; i < coding_book.size(); i++) {
        auto& yi = coding_book[i];

        for (int j = 0; j < i; j++) {
            auto& yj = coding_book[j];
            double dis2 = 0;

            for (int idx = 0; idx < yi.size(); idx++) {
                double res = yi[idx] - yj[idx];
                dis2 += res * res;
            }

            double res = distance_matrix[i][j] - std::sqrt(dis2);
            loss += res * res;
        }
    }
    return loss;
}

void compute_adjacency_table(vector<vector<double>>& distance_matrix, vector<vector<int>>& adjacency_table, const int K) {
    LOG(INFO) << "KNN: K = " << K << endl;

    const int node_number = distance_matrix.size();
    adjacency_table.resize(node_number, vector<int>(K));

    for (int i = 0; i < distance_matrix.size(); i++) {
        auto&   dis = distance_matrix[i];
        vector<int>     index(node_number);

        std::iota(index.begin(), index.end(), 0);   // 0, 1, 2, 3...
        std::partial_sort(index.begin(), index.begin()+K+1, index.end(), [&](const int& a, const int& b) {
            return (dis[a] < dis[b]);
        });

        std::copy(index.begin()+1, index.begin()+K+1, adjacency_table[i].begin());
        // (i, i) = 0.0 is the smalles value
    }

    LOG(INFO) << "success." << endl;
}

void embedding_optimization(vector<vector<double>>& coding_book, vector<vector<double>>& distance_matrix, vector<vector<int>>& adjacency_table) {
    const int node_number = coding_book.size();
    const int embedding_length = EMBEDDING_LENGTH;

    CHECK(node_number != 0) << "number of nodes equal 0" << endl;
    CHECK(coding_book[0].size() == embedding_length) << "length inconsistency" << endl;
    CHECK(distance_matrix.size() == node_number) << "distance matrix node number inconsistency" << endl;
    CHECK(distance_matrix[0].size() == node_number) << "distance matrix is not square" << endl;

    Problem problem;
    double *yl, *yi, *yj;
    const double w_lo = 1.0, w_dis = 1.0;

    for (int l = 0; l < node_number; l++) {
        yl = coding_book[l].data();
        auto& adjacency = adjacency_table[l];

        for (int idx_i = 0; idx_i < adjacency.size(); idx_i++) {
            auto& i = adjacency[idx_i];
            yi = coding_book[i].data();

            for (int idx_j = 0; idx_j < idx_i; idx_j++) {
                auto& j = adjacency[idx_j];
                yj = coding_book[j].data();

                problem.AddResidualBlock(
                    new AutoDiffCostFunction<LEAutoDiffCost, 1, EMBEDDING_LENGTH, EMBEDDING_LENGTH, EMBEDDING_LENGTH>(
                        new LEAutoDiffCost(w_lo)
                    ), NULL, yi, yj, yl
                );
            }
        }
    }

    for (int i = 0; i < node_number; i++) {
        for (int j = 0; j < i; j++) {
            auto& d = distance_matrix[i][j];
            yi = coding_book[i].data();
            yj = coding_book[j].data();

            problem.AddResidualBlock(
                new AutoDiffCostFunction<DistanceAutoDiffCost, 1, EMBEDDING_LENGTH, EMBEDDING_LENGTH>(
                    new DistanceAutoDiffCost(w_dis, d)
                ), NULL, yi, yj
            );
        }
    }

    Solver::Options opts;
    opts.max_num_iterations = 50;
    opts.minimizer_progress_to_stdout = true;

    Solver::Summary summary;
    Solve(opts, &problem, &summary);
    LOG(INFO) << summary.BriefReport() << endl;
}

void optimize_coding_book(const char * coding_book_fn, const char * distance_matrix_fn, const char * export_coding_book_fn) {
    const int K = 8;
    vector<vector<double>> coding_book, distance_matrix;
    vector<vector<int>> adjacency_table;


    import_coding_book(coding_book_fn, coding_book);
    import_distance_matrix(distance_matrix_fn, distance_matrix);
    compute_adjacency_table(distance_matrix, adjacency_table, K);

    double prev_distance_loss = compute_distance_loss(coding_book, distance_matrix);
    double prev_lo_loss = compute_LO_loss(coding_book, adjacency_table);
    LOG(INFO) << "previous distance loss = " << prev_distance_loss << endl;
    LOG(INFO) << "previous lo loss = " << prev_lo_loss << endl;

    embedding_optimization(coding_book, distance_matrix, adjacency_table);

    double after_distance_loss = compute_distance_loss(coding_book, distance_matrix);
    double after_lo_loss = compute_LO_loss(coding_book, adjacency_table);

    LOG(INFO) << "after distance loss = " << after_distance_loss << endl;
    LOG(INFO) << "after lo loss = " << after_lo_loss << endl;

    export_coding_book(export_coding_book_fn, coding_book);
}
}

int main(int argc, char** argv) {
    /*
        argv[1] - fn of coding book
        argv[2] - fn of distance matrix
        argv[3] - fn for optimized coding book
     */
    FLAGS_log_dir = "./log";
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    lass::optimize_coding_book(argv[1], argv[2], argv[3]);
}
