

#include <iostream>
#include <Eigen/Dense>

int main() {


    Eigen::MatrixXd xlocal(3, 3);

    xlocal <<
        0, 0, 1,
        0, 1, 0,
        0, 0, 0;

    //{
    //    0.5, 0.0, 0.0,
    //        0.5, 0.5, 0.0,
    //        0.0, 0.5, 0.0
    //},

    double xi = 0.5;
    double eta = 0.0;
    double zeta = 0.0;

    auto N_xi = std::vector<double>{ 1 - xi - eta, xi, eta };
    Eigen::MatrixXd DN_xi(2, 3);
    DN_xi << -1.0, 1.0, 0.0,
        -1.0, 0.0, 1.0;


    Eigen::Matrix<double, 3, 2> dx_xi = xlocal * DN_xi.transpose();

    Eigen::Vector3d normal_vector = dx_xi.col(0).cross(dx_xi.col(1));

    std::cout << normal_vector << std::endl;

}