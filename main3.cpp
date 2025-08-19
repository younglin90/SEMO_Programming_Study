#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cassert>
#include <functional>

#include <random>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>




inline static std::array <std::array<double, 3>, 27> xyz_switch_27 = {
	std::array<double, 3>{ 0.0, 0.0, 0.0},    // 0
	std::array<double, 3>{ 1.0, 0.0, 0.0},    // 1
	std::array<double, 3>{ 0.0, 1.0, 0.0},    // 2
	std::array<double, 3>{ -1.0, 0.0, 0.0},   // 3
	std::array<double, 3>{ 0.0, -1.0, 0.0},   // 4
	std::array<double, 3>{ 0.0, 0.0, 1.0},    // 5
	std::array<double, 3>{ 0.0, 0.0, -1.0},   // 6
	std::array<double, 3>{ 1.0, 1.0, 0.0},    // 7
	std::array<double, 3>{ -1.0, 1.0, 0.0},   // 8
	std::array<double, 3>{ -1.0, -1.0, 0.0},  // 9
	std::array<double, 3>{ 1.0, -1.0, 0.0},   // 10
	std::array<double, 3>{ 1.0, 0.0, 1.0},    // 11
	std::array<double, 3>{ 0.0, 1.0, 1.0},    // 12
	std::array<double, 3>{ -1.0, 0.0, 1.0},   // 13
	std::array<double, 3>{ 0.0, -1.0, 1.0},   // 14
	std::array<double, 3>{ 1.0, 0.0, -1.0},   // 15
	std::array<double, 3>{ 0.0, 1.0, -1.0},   // 16
	std::array<double, 3>{ -1.0, 0.0, -1.0},  // 17
	std::array<double, 3>{ 0.0, -1.0, -1.0},  // 18
	std::array<double, 3>{ 1.0, 1.0, 1.0},    // 19
	std::array<double, 3>{ -1.0, 1.0, 1.0},   // 20
	std::array<double, 3>{ -1.0, -1.0, 1.0},  // 21
	std::array<double, 3>{ 1.0, -1.0, 1.0},   // 22
	std::array<double, 3>{ 1.0, 1.0, -1.0},   // 23
	std::array<double, 3>{ -1.0, 1.0, -1.0},  // 24
	std::array<double, 3>{ -1.0, -1.0, -1.0}, // 25
	std::array<double, 3>{ 1.0, -1.0, -1.0}   // 26
};



int main() {


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(0.1, 1.0);



	double R = 1.0;
	double cv = 2.5;
	double rho = 1.0;// dis(gen);
	double u = 0.05;// dis(gen);
	double v = 0.05;//dis(gen);
	double w = 0.05;//dis(gen);
	double T = 0.1;// dis(gen);
	
	double usqr = u * u + v * v + w * w;

	double G0 = rho;
	double Gx = rho * u;
	double Gy = rho * v;
	double Gz = rho * w;
	double Gxx = rho * (u * u + R * T);
	double Gxy = rho * (u * v);
	double Gxz = rho * (u * w);
	double Gyy = rho * (v * v + R * T);
	double Gyz = rho * (v * w);
	double Gzz = rho * (w * w + R * T);
	double Gxxx = rho * (u * u * u + R * T * (3.0 * u));
	double Gxxy = rho * (u * u * v + R * T * (v));
	double Gxxz = rho * (u * u * w + R * T * (w));
	double Gxyy = rho * (u * v * v + R * T * (u));
	double Gxyz = rho * (u * v * w);
	double Gxzz = rho * (u * w * w + R * T * (u));
	double Gyyy = rho * (v * v * v + R * T * (3.0 * v));
	double Gyyz = rho * (v * v * w + R * T * (w));
	double Gyzz = rho * (v * w * w + R * T * (v));
	double Gzzz = rho * (w * w * w + R * T * (3.0 * w));


	//double G0 = rho * (cv * T + 0.5 * usqr);
	//double Gx = rho * u * ((R + cv) * T + 0.5 * usqr);
	//double Gy = rho * v * ((R + cv) * T + 0.5 * usqr);
	//double Gz = rho * w * ((R + cv) * T + 0.5 * usqr);
	//double Gxx = ((R + cv) * T + 0.5 * usqr) * rho * (u * u + R * T) + rho * R * T * u * u;
	//double Gxy = ((R + cv) * T + 0.5 * usqr) * rho * (u * v) + rho * R * T * u * v;
	//double Gxz = ((R + cv) * T + 0.5 * usqr) * rho * (u * w) + rho * R * T * u * w;
	//double Gyy = ((R + cv) * T + 0.5 * usqr) * rho * (v * v + R * T) + rho * R * T * v * v;
	//double Gyz = ((R + cv) * T + 0.5 * usqr) * rho * (v * w) + rho * R * T * v * w;
	//double Gzz = ((R + cv) * T + 0.5 * usqr) * rho * (w * w + R * T) + rho * R * T * w * w;

	const int N = 20;
	Eigen::Vector<double, N> lambda;
	Eigen::Vector<double, N> Gmat;
	Eigen::Matrix<double, N, N> jacobian;


	lambda.setZero();

	int iter = 0;
	while (iter < 150) {
		++iter;

		Gmat.setZero();
		jacobian.setZero();
		for (int i = 0; i < 27; ++i) {

			const auto& cxyz = xyz_switch_27[i];

			Eigen::Vector<double, N> xi_tmp;
			xi_tmp[0] = 1.0;
			xi_tmp[1] = cxyz[0];
			xi_tmp[2] = cxyz[1];
			xi_tmp[3] = cxyz[2];
			xi_tmp[4] = cxyz[0] * cxyz[0];
			xi_tmp[5] = cxyz[0] * cxyz[1];
			xi_tmp[6] = cxyz[0] * cxyz[2];
			xi_tmp[7] = cxyz[1] * cxyz[1];
			xi_tmp[8] = cxyz[1] * cxyz[2];
			xi_tmp[9] = cxyz[2] * cxyz[2];
			xi_tmp[10] = cxyz[0] * cxyz[0] * cxyz[0];
			xi_tmp[11] = cxyz[0] * cxyz[0] * cxyz[1];
			xi_tmp[12] = cxyz[0] * cxyz[0] * cxyz[2];
			xi_tmp[13] = cxyz[0] * cxyz[1] * cxyz[1];
			xi_tmp[14] = cxyz[0] * cxyz[1] * cxyz[2];
			xi_tmp[15] = cxyz[0] * cxyz[2] * cxyz[2];
			xi_tmp[16] = cxyz[1] * cxyz[1] * cxyz[1];
			xi_tmp[17] = cxyz[1] * cxyz[1] * cxyz[2];
			xi_tmp[18] = cxyz[1] * cxyz[2] * cxyz[2];
			xi_tmp[19] = cxyz[2] * cxyz[2] * cxyz[2];

			double feq_i_coeff = 0.0;
			for (int j = 0; j < N; ++j) {
				feq_i_coeff += lambda[j] * xi_tmp[j];
			}
			double feq_i = rho * std::exp(-(1.0 + feq_i_coeff));

			for (int j = 0; j < N; ++j) {
				Gmat[j] += feq_i * xi_tmp[j];

			}

			for (int j = 0; j < N; ++j) {
				for (int k = 0; k < N; ++k) {
					jacobian(j, k) -= feq_i * xi_tmp[j] * xi_tmp[k];
				}
			}
		}
		Gmat[0] -= G0;
		Gmat[1] -= Gx;
		Gmat[2] -= Gy;
		Gmat[3] -= Gz;
		Gmat[4] -= Gxx;
		Gmat[5] -= Gxy;
		Gmat[6] -= Gxz;
		Gmat[7] -= Gyy;
		Gmat[8] -= Gyz;
		Gmat[9] -= Gzz;
		Gmat[10] -= Gxxx;
		Gmat[11] -= Gxxy;
		Gmat[12] -= Gxxz;
		Gmat[13] -= Gxyy;
		Gmat[14] -= Gxyz;
		Gmat[15] -= Gxzz;
		Gmat[16] -= Gyyy;
		Gmat[17] -= Gyyz;
		Gmat[18] -= Gyzz;
		Gmat[19] -= Gzzz;

		auto dlambda = jacobian.fullPivLu().solve(Gmat);
		//Eigen::LeastSquaresConjugateGradient<Eigen::MatrixX<double>> lscg;
		//lscg.compute(jacobian);
		//auto dlambda = lscg.solve(Gmat);
		//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg;
		//cg.compute(jacobian);
		//Eigen::Vector<double, N> dlambda = cg.solve(Gmat);

		double norm = dlambda.squaredNorm();
		std::cout << norm << std::endl;
		if (norm < 1.e-12) break;


		lambda -= dlambda;
	}


	std::cout << lambda.transpose() << std::endl;


	return 0;

}