
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <finite-diff/finitediff.hpp>
#include <autodiff/autodiff_types.hpp>


double point_point_distance(
	const Eigen::Ref<const Eigen::Vector3d>& p0,
	const Eigen::Ref<const Eigen::Vector3d>& p1)
{
	return (p1 - p0).squaredNorm();
}

double point_line_distance(
	const Eigen::Ref<const Eigen::Vector3d>& p,
	const Eigen::Ref<const Eigen::Vector3d>& e0,
	const Eigen::Ref<const Eigen::Vector3d>& e1)
{
	return (e0 - p).head<3>().cross((e1 - p).head<3>()).squaredNorm()
		/ (e1 - e0).squaredNorm();
}


template<typename T>
T point_plane_distance(
	const Eigen::Vector3<T>& p,
	const Eigen::Vector3<T>& origin,
	const Eigen::Vector3<T>& normal)
{
	const T point_to_plane = (p - origin).dot(normal);
	return point_to_plane * point_to_plane / normal.squaredNorm();
}

template<typename T>
T point_plane_distance(
	const Eigen::Vector3<T>& p,
	const Eigen::Vector3<T>& t0,
	const Eigen::Vector3<T>& t1,
	const Eigen::Vector3<T>& t2)
{
	return point_plane_distance(p, t0, (t1 - t0).cross(t2 - t0));
}


template<typename T>
T line_line_distance(
	const Eigen::Vector3<T>& ea0,
	const Eigen::Vector3<T>& ea1,
	const Eigen::Vector3<T>& eb0,
	const Eigen::Vector3<T>& eb1)
{
	const Eigen::Vector3<T> normal = (ea1 - ea0).cross(eb1 - eb0);
	const T line_to_line = (eb0 - ea0).dot(normal);
	return line_to_line * line_to_line / normal.squaredNorm();
}


template<typename T>
T barrier(const T& d, const double& dhat)
{
	if (d <= 0.0) {
		return T(1.e100);
	}
	if (d >= dhat) {
		return T(0);
	}
	const T d_minus_dhat = (d - dhat);
	return -d_minus_dhat * d_minus_dhat * log(d / dhat);
}



int main() {

	Eigen::Vector3d v0, v1, v2, v3;
	double kappa = 5.67918e+09;
	double dhat = 1.e-3;

	double dist2;
	//case1 edge-edge -> 통과
	//{
	//	v0 << 2.10212, -1.999, 0.618025;
	//	v1 << 2.2, - 1.999, - 2.4436e-17;
	//	v2 << 2.2, - 2, 1e-06;
	//	v3 << 2.30557, - 1.44721, 0.324916;
	//	dist2 = line_line_distance<double>(v0, v1, v2, v3);
	//}
	//case2 edge-edge
	//{
	//	v0 << 2.2, - 1.999, - 2.4436e-17;
	//	v1 << 2.09443 ,- 2.55179, - 0.324915;
	//	v2 << 2.30557, - 1.44721, - 0.324915;
	//	v3 << 2.2, - 2, 1e-06;
	//	dist2 = line_line_distance<double>(v0, v1, v2, v3);
	//}
	//case3 face-vertex
	{
		v0 << 2.2, - 1.999, - 2.4436e-17;
		v1 << 2.2, - 2, 1e-06;
		v2 << 2.30557, - 1.44721, 0.324916;
		v3 << 2.30557, - 1.44721, - 0.324915;
	}

	Eigen::VectorXd v12(12);
	v12.segment(0, 3) = v0;
	v12.segment(3, 3) = v1;
	v12.segment(6, 3) = v2;
	v12.segment(9, 3) = v3;

	using Diff = ipc::rigid::AutodiffType<12>;
	Diff::activate();
	auto x_autodiff = Diff::d2vars(0, v12);
	auto dist2_autodiff = point_plane_distance<Diff::DDouble2>(
		x_autodiff.segment(0, 3), x_autodiff.segment(3, 3), x_autodiff.segment(6, 3), x_autodiff.segment(9, 3));


	//std::cout << point_point_distance(v0, v2) << std::endl;
	//std::cout << point_point_distance(v0, v3) << std::endl;
	//std::cout << point_point_distance(v1, v2) << std::endl;
	//std::cout << point_point_distance(v1, v3) << std::endl;
	//std::cout << point_line_distance(v0, v2, v3) << std::endl;
	//std::cout << point_line_distance(v1, v2, v3) << std::endl;
	//std::cout << point_line_distance(v2, v0, v1) << std::endl;
	//std::cout << point_line_distance(v3, v0, v1) << std::endl;
	//std::cout << line_line_distance(v0, v1,v2,v3) << std::endl;

	//std::cout << point_point_distance(v0, v1) << std::endl;
	//std::cout << point_point_distance(v0, v2) << std::endl;
	//std::cout << point_point_distance(v0, v3) << std::endl;
	//std::cout << point_line_distance(v0, v1, v2) << std::endl;
	//std::cout << point_line_distance(v0, v2, v3) << std::endl;
	//std::cout << point_line_distance(v0, v3, v1) << std::endl;
	//std::cout << point_plane_distance(v0, v1,v2,v3) << std::endl;

	auto energy_barrier = kappa * barrier<Diff::DDouble2>(dist2_autodiff, dhat * dhat);

	std::cout << energy_barrier.getValue() << std::endl;
	std::cout << energy_barrier.getGradient() << std::endl;
	std::cout << energy_barrier.getHessian() << std::endl;

	/*
	0.0127807

	 0.0396337  -0.011259 0.00627727    48.4998   -13.7776     7.6815   -48.4582    13.7658    -7.6749 -0.0813145  0.0230995 -0.0128787

	   0.182014  -0.0140121   -0.035302     149.673     3.60714      -54.77    -149.567    -3.59757     54.7193   -0.287855  0.00443878   0.0859804
	 -0.0140121 -0.00672731   0.0159984    -42.5562    -1.01399     15.5528     42.5601     1.00161    -15.5331   0.0100753   0.0191065  -0.0357805
	  -0.035302   0.0159984  -0.0157482     23.7697    0.553089    -8.66442    -23.6888    -0.56979     8.66655  -0.0455911 0.000703022   0.0136177
		149.673    -42.5562     23.7697      154927    -44057.1     24616.1     -154794     44019.4    -24595.1    -282.309     80.2745    -44.8443
		3.60714    -1.01399    0.553089    -44057.1     12528.7    -7000.15     44061.1    -12529.8     7000.75    -7.53898     2.11967    -1.15666
		 -54.77     15.5528    -8.66442     24616.1    -7000.15     3911.18    -24516.6     6971.88    -3895.41    -44.7127      12.714    -7.10252
	   -149.567     42.5601    -23.6888     -154794     44061.1    -24516.6      154662    -44023.3     24495.7     282.048    -80.2701     44.6713
	   -3.59757     1.00161    -0.56979     44019.4    -12529.8     6971.88    -44023.3     12530.9     -6972.5     7.53777    -2.09951     1.19385
		54.7193    -15.5331     8.66655    -24595.1     7000.75    -3895.41     24495.7     -6972.5     3879.67     44.6713    -12.7133     7.07513
	  -0.287855   0.0100753  -0.0455911    -282.309    -7.53898    -44.7127     282.048     7.53777     44.6713    0.549017 -0.00886442   0.0869543
	 0.00443878   0.0191065 0.000703022     80.2745     2.11967      12.714    -80.2701    -2.09951    -12.7133 -0.00886442  -0.0392687 -0.00140396
	  0.0859804  -0.0357805   0.0136177    -44.8443    -1.15666    -7.10252     44.6713     1.19385     7.07513   0.0869543 -0.00140396    0.013772
	*/

	//using Diff = ipc::rigid::AutodiffType<6>;
	//Diff::activate();
	//auto x_autodiff = Diff::d2vars(0, dofs);
	//auto d_autodiff = (x_autodiff.segment(0, 3) - x_autodiff.segment(3, 3)).norm() - r0 - r1;

	//Diff::DDouble2 b_autodff;
	//b_autodff = 0.0;
	//if (d_autodiff < dhat) {
	//	b_autodff = -(d_autodiff - dhat) * (d_autodiff - dhat) * log(d_autodiff / dhat);
	//}
	//Diff::DDouble2 energy_autodff = kappa * b_autodff;
	//const auto& grad_autodiff = energy_autodff.getGradient();
	//const auto& hess_autodiff = energy_autodff.getHessian();


	//std::cout << grad_autodiff << std::endl;
	//std::cout << hess_autodiff << std::endl;
	//


}


//
//double compute_objective(
//	Eigen::VectorXd& dofs,
//	Eigen::VectorXd& Ddofs,
//	double& r0, double& r1, double& dhat, double& dt,
//	double& mass0, double& mass1, double& kappa,
//	Eigen::VectorXd& grad,
//	Eigen::SparseMatrix<double>& hess
//) {
//
//	Eigen::Vector3d x0 = dofs.segment(0, 3);
//	Eigen::Vector3d v0 = Ddofs.segment(0, 3);
//	Eigen::Vector3d x1 = dofs.segment(3, 3);
//	Eigen::Vector3d v1 = Ddofs.segment(3, 3);
//
//	double d = (x0 - x1).norm() - r0 - r1;
//
//	double b = 0.0;
//	if (d < dhat) {
//		b = -(d - dhat) * (d - dhat) * std::log(d / dhat);
//	}
//
//	Eigen::Vector3d x0_hat = x0 + v0 * dt;
//	Eigen::Vector3d x1_hat = x1 + v1 * dt;
//
//	double energy = (mass0 / dt / dt) * (x0 - x0_hat).dot(x0 - x0_hat) +
//		(mass1 / dt / dt) * (x1 - x1_hat).dot(x1 - x1_hat) +
//		kappa * b;
//
//	auto calc_energy = [&Ddofs = Ddofs, r0, r1, dhat, mass0, mass1, dt, kappa](const Eigen::VectorXd x)->double {
//
//		Eigen::Vector3d x0 = x.segment(0, 3);
//		Eigen::Vector3d v0 = Ddofs.segment(0, 3);
//		Eigen::Vector3d x1 = x.segment(3, 3);
//		Eigen::Vector3d v1 = Ddofs.segment(3, 3);
//
//		double d = (x0 - x1).norm() - r0 - r1;
//
//		double b = 0.0;
//		if (d < dhat && d > 0.0) {
//			b = -(d - dhat) * (d - dhat) * std::log(d / dhat);
//		}
//
//		Eigen::Vector3d x0_hat = x0 + v0 * dt;
//		Eigen::Vector3d x1_hat = x1 + v1 * dt;
//
//		double energy = 
//			0.5 * (mass0 / dt / dt) * (x0 - x0_hat).dot(x0 - x0_hat) +
//			0.5 * (mass1 / dt / dt) * (x1 - x1_hat).dot(x1 - x1_hat) +
//			kappa * b;
//
//		return energy;
//
//		};
//
//	Eigen::VectorXd grad_fd;
//	Eigen::MatrixXd hess_fd;
//	fd::finite_gradient(dofs, calc_energy, grad_fd);
//	fd::finite_hessian(dofs, calc_energy, hess_fd);
//	std::cout << grad_fd << std::endl;
//	std::cout << hess_fd << std::endl;
//
//	using Diff = ipc::rigid::AutodiffType<6>;
//	Diff::activate();
//	auto x_autodiff = Diff::d2vars(0, dofs);
//	auto d_autodiff = (x_autodiff.segment(0, 3) - x_autodiff.segment(3, 3)).norm() - r0 - r1;
//
//	Diff::DDouble2 b_autodff;
//	b_autodff = 0.0;
//	if (d_autodiff < dhat) {
//		b_autodff = -(d_autodiff - dhat) * (d_autodiff - dhat) * log(d_autodiff / dhat);
//	}
//	Diff::DDouble2 energy_autodff =
//		0.5 * (mass0 / dt / dt) * (x_autodiff.segment(0, 3) - x0_hat).dot(x_autodiff.segment(0, 3) - x0_hat) +
//		0.5 * (mass1 / dt / dt) * (x_autodiff.segment(3, 3) - x1_hat).dot(x_autodiff.segment(3, 3) - x1_hat) +
//		kappa * b_autodff;
//	const auto& grad_autodiff = energy_autodff.getGradient();
//	const auto& hess_autodiff = energy_autodff.getHessian();
//
//	//energy = energy_autodff.getValue();
//	//grad += grad_autodiff;
//	//for (int i = 0; i < 6; ++i) {
//	//	for (int j = 0; j < 6; ++j) {
//	//		hess.coeffRef(i, j) += hess_autodiff(i, j);
//	//	}
//	//}
//
//	std::cout << grad_autodiff << std::endl;
//	std::cout << hess_autodiff << std::endl;
//
//
//
//	Eigen::Vector3d dddx0 = (x0 - x1) / (x0 - x1).norm();
//	Eigen::Vector3d dddx1 = -(x0 - x1) / (x0 - x1).norm();
//
//	Eigen::VectorXd grad_numeric(6);
//	grad_numeric.segment(0, 3) = (mass0 / dt / dt) * (x0 - x0_hat) +
//		(kappa * (d - dhat) * (-2.0 * std::log(d / dhat) - 1.0 + dhat / d)) * dddx0;
//	grad_numeric.segment(3, 3) = (mass1 / dt / dt) * (x1 - x1_hat) +
//		(kappa * (d - dhat) * (-2.0 * std::log(d / dhat) - 1.0 + dhat / d)) * dddx1;
//	Eigen::MatrixXd hess_numeric(6, 6);
//	hess_numeric.setZero();
//	hess_numeric.diagonal().segment(0, 3).setConstant(mass0 / dt / dt);
//	hess_numeric.diagonal().segment(3, 3).setConstant(mass1 / dt / dt);
//	double coeff1 = (kappa * (-2.0 * std::log(d / dhat) - 1.0 + dhat / d) +
//		kappa * (d - dhat) * (-2.0 / d - dhat / d / d));
//	hess_numeric.block(0, 0, 3, 3) += coeff1 * dddx0 * dddx0.transpose();
//	hess_numeric.block(0, 3, 3, 3) += coeff1 * dddx0 * dddx1.transpose();
//	hess_numeric.block(3, 0, 3, 3) += coeff1 * dddx1 * dddx0.transpose();
//	hess_numeric.block(3, 3, 3, 3) += coeff1 * dddx1 * dddx1.transpose();
//
//	std::cout << grad_numeric << std::endl;
//	std::cout << hess_numeric << std::endl;
//
//	grad += grad_numeric;
//	for (int i = 0; i < 6; ++i) {
//		for (int j = 0; j < 6; ++j) {
//			hess.coeffRef(i, j) += hess_numeric(i, j);
//		}
//	}
//
//
//	return energy;
//
//
//
//}
//
//int main() {
//
//	Eigen::SparseMatrix<double> hess;
//	hess.resize(6, 6);
//
//	Eigen::VectorXd grad;
//	grad.resize(6);
//
//
//	Eigen::VectorXd dofs, Ddofs;
//	dofs.resize(6);
//	dofs.segment(0, 3) = Eigen::Vector3d(0, 0, 0);
//	dofs.segment(3, 3) = Eigen::Vector3d(2.0000005, 0, 0);
//	Ddofs.resize(6);
//	Ddofs.segment(0, 3) = Eigen::Vector3d(10, 0, 0);
//	Ddofs.segment(3, 3) = Eigen::Vector3d(0, 0, 0);
//
//	double dt = 0.1;
//	double dhat = 0.001;
//	double kappa = 1.e3;
//	double r0 = 1.0;
//	double r1 = 1.0;
//	double mass0 = 1.0;
//	double mass1 = 1.0;
//
//	hess.setZero();
//	grad.setZero();
//
//	double energy = compute_objective(
//			dofs, Ddofs, r0, r1, dhat, dt, mass0, mass1, kappa, grad, hess);
//
//	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver(hess);
//	Eigen::VectorXd d_dofs = solver.solve(-grad);
//
//	std::cout << d_dofs << std::endl;
//
//	//double step_length = 1.0;
//
//	//double energy0 = energy;
//	//for (int iter = 0; iter < 1; ++iter) {
//
//	//	Eigen::VectorXd dofs_tmp = dofs + step_length * d_dofs;
//	//	double energy_tmp = compute_objective(
//	//		dofs_tmp, Ddofs, r0, r1, dhat, dt, mass0, mass1, kappa, grad, hess);
//
//	//	std::cout << energy_tmp << " " << energy0 << std::endl;
//
//	//	if (energy_tmp < energy0) break;
//
//	//	step_length *= 0.9;
//	//}
//
//	//dofs = dofs + step_length * d_dofs;
//
//
//	//std::cout << dofs << std::endl;
//
//
//}
