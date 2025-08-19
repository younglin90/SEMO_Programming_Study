
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <finite-diff/finitediff.hpp>
#include "./3rd-party/autodiff/reverse/var.hpp"
#include "./3rd-party/autodiff/reverse/var/eigen.hpp"

Eigen::VectorXd compute_shape_function_3x1(
	const double& r, const double& s
) {
	return (Eigen::VectorXd(3) << 1 - r - s, r, s).finished();
}

Eigen::MatrixXd compute_derivative_shape_function_2x3(
	const double& r, const double& s
) {
	return (Eigen::MatrixXd(2, 3) <<
		-1.0, 1.0, 0.0,
		-1.0, 0.0, 1.0).finished();
}

template<typename T>
Eigen::MatrixX<T> compute_derivative_position_3x3(
	const double& t,
	const Eigen::MatrixX<T>& xlocal,
	const Eigen::VectorXd& thicklocal,
	const Eigen::MatrixX<T>& Vnlocal,
	const Eigen::MatrixXd& DN_xi,
	const Eigen::VectorXd& N_xi
) {

	Eigen::MatrixX<T> Dx_xi(3, 3);
	Dx_xi.setZero();
	Dx_xi.block(0, 0, 3, 2) = xlocal * DN_xi.cast<T>().transpose();
	for (int i = 0; i < 3; ++i) {
		for (int xidim = 0; xidim < 2; ++xidim) {
			Dx_xi(0, xidim) += 0.5 * t * (thicklocal[i] * DN_xi(xidim, i)) * Vnlocal(0, i);
			Dx_xi(1, xidim) += 0.5 * t * (thicklocal[i] * DN_xi(xidim, i)) * Vnlocal(1, i);
			Dx_xi(2, xidim) += 0.5 * t * (thicklocal[i] * DN_xi(xidim, i)) * Vnlocal(2, i);
		}
		Dx_xi(0, 2) += 0.5 * (thicklocal[i] * N_xi[i]) * Vnlocal(0, i);
		Dx_xi(1, 2) += 0.5 * (thicklocal[i] * N_xi[i]) * Vnlocal(1, i);
		Dx_xi(2, 2) += 0.5 * (thicklocal[i] * N_xi[i]) * Vnlocal(2, i);
	}
	return Dx_xi;

}

Eigen::Matrix<double, 2, 3> get_tangent_vectors(const Eigen::Vector3d& Vn)
{
	// 기준 벡터 e2 설정
	Eigen::Vector3d e2(0.0, 1.0, 0.0);

	Eigen::Vector3d cross_e2_Vn = e2.cross(Vn);

	Eigen::Vector3d V1n;

	// cross product의 크기가 매우 작으면(벡터가 평행하면) 다른 기준 벡터를 사용
	if (cross_e2_Vn.norm() < 1e-8) {
		// e2와 평행하므로 다른 기준벡터 사용
		Eigen::Vector3d alt_e(1.0, 0.0, 0.0);
		V1n = alt_e.cross(Vn).normalized();
	}
	else {
		// 일반적인 경우
		V1n = cross_e2_Vn.normalized();
	}

	// 두 번째 직교벡터 계산
	Eigen::Vector3d V2n = Vn.cross(V1n).normalized();

	// 결과를 행렬 형태로 반환
	Eigen::Matrix<double, 2, 3> result;
	result.row(0) = V1n;
	result.row(1) = V2n;

	return result;
}


static Eigen::Matrix3d cauchy_stress(const double& mu_, const double& lambda_, const double& J, const Eigen::Matrix3d& b) {
	double logJ = std::log(J);
	auto cons_I = Eigen::Matrix3d::Identity();
	return mu_ / J * (b - cons_I) + (lambda_ * logJ) / J * cons_I;
}


static Eigen::MatrixXd elasticity_modulus_Voigt(
	const double& mu_, const double& lambda_, const double& J
) {
	double logJ = std::log(J);
	double mu_prime = (mu_ - lambda_ * logJ) / J;
	double lambda_prime = lambda_ / J;
	Eigen::MatrixXd modulus = Eigen::MatrixXd::Zero(6, 6);
	modulus.coeffRef(0, 0) = lambda_prime + 2.0 * mu_prime;
	modulus.coeffRef(0, 1) = lambda_prime;
	modulus.coeffRef(0, 2) = lambda_prime;
	modulus.coeffRef(1, 0) = lambda_prime;
	modulus.coeffRef(1, 1) = lambda_prime + 2.0 * mu_prime;
	modulus.coeffRef(1, 2) = lambda_prime;
	modulus.coeffRef(2, 0) = lambda_prime;
	modulus.coeffRef(2, 1) = lambda_prime;
	modulus.coeffRef(2, 2) = lambda_prime + 2.0 * mu_prime;

	modulus.coeffRef(3, 3) = mu_prime;
	modulus.coeffRef(4, 4) = mu_prime;
	modulus.coeffRef(5, 5) = mu_prime;

	return modulus;
}


int main() {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist1(0.0, 0.5);
	std::uniform_real_distribution<double> dist2(-1.0, 1.0);
	std::uniform_real_distribution<double> dist3(0.1, 5.0);
	std::uniform_real_distribution<double> dist4(1.e2, 1.e8);

	int ivs_size = 3;

	double mu = 7142.86;// dist4(gen);
	double lambda = 28571.43;// dist4(gen);

	Eigen::VectorXd thicklocal(3);
	thicklocal.setConstant(0.01);

	Eigen::MatrixXd Vn0local = Eigen::MatrixXd::NullaryExpr(3, 3, [&]() {
		return dist2(gen);
		});
	Vn0local.col(0).normalize();
	Vn0local.col(1).normalize();
	Vn0local.col(2).normalize();
	//Eigen::MatrixXd Vn0local(3, 3);
	//Vn0local.col(0) << 1.0, 0.0, 0.0;
	//Vn0local.col(1) << 1.0, 0.0, 0.0;
	//Vn0local.col(2) << 1.0, 0.0, 0.0;

	//Eigen::MatrixXd Vnlocal = Eigen::MatrixXd::NullaryExpr(3, 3, [&]() {
	//	return dist2(gen);
	//	});
	//Vnlocal.col(0).normalize();
	//Vnlocal.col(1).normalize();
	//Vnlocal.col(2).normalize();
	Eigen::MatrixXd Vnlocal = Vn0local;

	Eigen::MatrixXd x0local = Eigen::MatrixXd::NullaryExpr(3, 3, [&]() {
		return dist2(gen);
		});
	//Eigen::MatrixXd xlocal = Eigen::MatrixXd::NullaryExpr(3, 3, [&]() {
	//	return dist2(gen);
	//});


	auto xi = dist1(gen);
	auto xj = dist1(gen);
	auto xk = dist2(gen);


	Eigen::VectorXd dofs(5 * ivs_size); 
	{
		Eigen::MatrixXd xlocal = x0local;
		Eigen::Vector3d cen = 0.333333333 * (xlocal.col(0) + xlocal.col(1) + xlocal.col(2));
		xlocal.col(0) = cen + (xlocal.col(0) - cen) * dist3(gen);
		xlocal.col(1) = cen + (xlocal.col(1) - cen) * dist3(gen);
		xlocal.col(2) = cen + (xlocal.col(2) - cen) * dist3(gen);


		dofs.segment(0, 3) = xlocal.col(0);
		dofs.segment(5, 3) = xlocal.col(1);
		dofs.segment(10, 3) = xlocal.col(2);
		dofs.segment(3, 2).setZero();
		dofs.segment(8, 2).setZero();
		dofs.segment(13, 2).setZero();
	}



	//auto calc_phi = [&](const Eigen::VectorXd& dofs_in)->double {

	//	Eigen::MatrixXd Vnlocal(3, 3);

	//	// 쿼터니안 방법
	//	for (int anode = 0; anode < ivs_size; ++anode) {
	//		const auto& alpha = dofs_in(5 * anode + 3);
	//		const auto& beta = dofs_in(5 * anode + 4);
	//		const auto& Vn = Vnlocal.col(anode);
	//		auto tan_vec = get_tangent_vectors(Vn);
	//		Eigen::Quaterniond q1(Eigen::AngleAxisd(alpha, tan_vec.row(0)));
	//		Eigen::Quaterniond q2(Eigen::AngleAxisd(beta, tan_vec.row(1)));
	//		Vnlocal.col(anode) = (q2 * q1) * Vn;
	//		Vnlocal.col(anode).normalize();
	//	}

	//	Eigen::MatrixXd xlocal(3, 3);
	//	xlocal.col(0) = dofs_in.segment(0, 3);
	//	xlocal.col(1) = dofs_in.segment(5, 3);
	//	xlocal.col(2) = dofs_in.segment(10, 3);


	//	auto xi = dist1(gen);
	//	auto xj = dist1(gen);
	//	auto xk = dist2(gen);

	//	Eigen::VectorXd N_xi = compute_shape_function_3x1(xi, xj);
	//	Eigen::MatrixXd DN_xi = compute_derivative_shape_function_2x3(xi, xj);
	//	Eigen::MatrixXd Dx_xi = compute_derivative_position_3x3(xk, xlocal, thicklocal, Vnlocal, DN_xi, N_xi);
	//	Eigen::MatrixXd Dx0_xi = compute_derivative_position_3x3(xk, x0local, thicklocal, Vn0local, DN_xi, N_xi);
	//	DN_xi.conservativeResize(3, 3); // 3 x v_size

	//	DN_xi.bottomRows(1).setZero();

	//	Eigen::Matrix3d Dxi_x0 = Dx0_xi.inverse(); // {xi/x0, xi/y0, xi/z0} {eta/x0, eta/y0, eta/z0} {zeta/x0, zeta/y0, zeta/z0}
	//	Eigen::Matrix3d Dxi_x = Dx_xi.inverse(); // {xi/x, xi/y, xi/z} {eta/x, eta/y, eta/z} {zeta/x, zeta/y, zeta/z}

	//	Eigen::MatrixXd DN_x0 = Dxi_x0.transpose() * DN_xi; // 3 x v_size, {N1/x0, N2/x0, ...} {N1/y0, N2/y0, ...} {N1/z0, N2/z0, ...}
	//	Eigen::MatrixXd DN_x = Dxi_x.transpose() * DN_xi; // 3 x v_size, {N1/x, N2/x, ...} {N1/y, N2/y, ...} {N1/z, N2/z, ...}

	//	double Jx_xi = std::abs(Dx_xi.determinant());
	//	Eigen::Matrix3d F = xlocal * DN_x0.transpose();
	//	//------ shell
	//	for (int anode = 0; anode < ivs_size; ++anode) {
	//		const auto& Na = N_xi[anode];
	//		F.col(0) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 0) * Na + xk * DN_x0(0, anode)) * Vnlocal.col(anode);
	//		F.col(1) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 1) * Na + xk * DN_x0(1, anode)) * Vnlocal.col(anode);
	//		F.col(2) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 2) * Na + xk * DN_x0(2, anode)) * Vnlocal.col(anode);
	//	}

	//	double J = F.determinant();
	//	//std::cout << F << std::endl;
	//	//std::cout << J << std::endl;
	//	assert(J > 0.0);
	//	Eigen::Matrix3d b = F * F.transpose();


	//	double phi = 0.5 * mu * (b.trace() - 3.0) - mu * std::log(J) + 0.5 * lambda * std::log(J) * std::log(J);
	//	std::cout << phi << std::endl;

	//	return phi;

	//};



	//Eigen::VectorXd grad;
	//fd::finite_gradient(dofs, calc_phi, grad);

	//std::cout << grad << std::endl;
	

	auto calc_phi2 = [&](autodiff::VectorXvar dofs_in)->autodiff::var {

		Eigen::MatrixX<autodiff::var> Vnlocal_i(3, 3);
		Vnlocal_i.setZero();

		for (int anode = 0; anode < ivs_size; ++anode) {
			const auto& alpha = dofs_in(5 * anode + 3);
			const auto& beta = dofs_in(5 * anode + 4);
			Eigen::Vector3d Vn = Vnlocal.col(anode);
			auto tan_vec = get_tangent_vectors(Vn);
			Eigen::Vector3d tan0 = tan_vec.row(0);
			Eigen::Vector3d tan1 = tan_vec.row(1);
			//Eigen::Quaternion<autodiff::var> q1(Eigen::AngleAxis<autodiff::var>(alpha, tan_vec.row(0)));
			//Eigen::Quaternion<autodiff::var> q2(Eigen::AngleAxis<autodiff::var>(beta, tan_vec.row(1)));
			//Vnlocal.col(anode) = (q2 * q1) * Vn0;

			// European Journal of Mechanics / A Solids 89 (2021) 104283
			Eigen::Vector3<autodiff::var> theta = alpha * tan0 + beta * tan1;
			Eigen::Matrix3<autodiff::var> Theta;
			Theta << 0.0, -theta[2], theta[1],
				theta[2], 0.0, -theta[0],
				-theta[1], theta[0], 0.0;
			//autodiff::var theta_mag = theta[0] * theta[0];

			//Eigen::Matrix3<autodiff::var> Q;
			//Q.setIdentity();
			//Q += theta_mag * Theta;
			////Q += sin(theta_mag) / theta_mag * Theta;// +
			//	//0.5 * pow(sin(0.5 * theta_mag) / (0.5 * theta_mag), 2.0) * Theta * Theta;

			Vnlocal_i.col(anode) = 
				(Eigen::Matrix3<autodiff::var>::Identity() + Theta) * 
				Vnlocal.col(anode);

		}


		Eigen::MatrixX<autodiff::var> xlocal(3, 3);
		xlocal.col(0) = dofs_in.segment(0, 3);
		xlocal.col(1) = dofs_in.segment(5, 3);
		xlocal.col(2) = dofs_in.segment(10, 3);


		auto get_Dx_xi_tmp = [&xlocal = xlocal, &thicklocal = thicklocal,
			&Vnlocal_i = Vnlocal_i](double xi_tmp, double xj_tmp, double xk_tmp)->Eigen::MatrixX<autodiff::var> {
			Eigen::VectorXd N_xi = compute_shape_function_3x1(xi_tmp, xj_tmp);
			Eigen::MatrixXd DN_xi = compute_derivative_shape_function_2x3(xi_tmp, xj_tmp);
			Eigen::MatrixX<autodiff::var> Dx_xi = 
				compute_derivative_position_3x3(xk_tmp, xlocal, thicklocal, Vnlocal_i, DN_xi, N_xi);
			return Dx_xi;
		};
		auto get_Dx0_xi_tmp = [&x0local = x0local, &thicklocal = thicklocal,
			&Vn0local = Vn0local](double xi_tmp, double xj_tmp, double xk_tmp)->Eigen::MatrixX<double> {
			Eigen::VectorXd N_xi = compute_shape_function_3x1(xi_tmp, xj_tmp);
			Eigen::MatrixXd DN_xi = compute_derivative_shape_function_2x3(xi_tmp, xj_tmp);
			Eigen::MatrixX<double> Dx_xi =
				compute_derivative_position_3x3<double>(xk_tmp, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			return Dx_xi;
		};



		// A
		Eigen::MatrixX<double> Dx0_xi_A = get_Dx0_xi_tmp(1.0 / 6.0, 2.0 / 3.0, 0.0);
		Eigen::MatrixX<autodiff::var> Dx_xi_A = get_Dx_xi_tmp(1.0 / 6.0, 2.0 / 3.0, 0.0);
		// B
		Eigen::MatrixX<double> Dx0_xi_B = get_Dx0_xi_tmp(2.0 / 3.0, 1.0 / 6.0, 0.0);
		Eigen::MatrixX<autodiff::var> Dx_xi_B = get_Dx_xi_tmp(2.0 / 3.0, 1.0 / 6.0, 0.0);
		// C
		Eigen::MatrixX<double> Dx0_xi_C = get_Dx0_xi_tmp(1.0 / 6.0, 1.0 / 6.0, 0.0);
		Eigen::MatrixX<autodiff::var> Dx_xi_C = get_Dx_xi_tmp(1.0 / 6.0, 1.0 / 6.0, 0.0);
		// D
		Eigen::MatrixX<double> Dx0_xi_D = get_Dx0_xi_tmp(1.0 / 3.0 + 1.e-4, 1.0 / 3.0 - 2.e-4, 0.0);
		Eigen::MatrixX<autodiff::var> Dx_xi_D = get_Dx_xi_tmp(1.0 / 3.0 + 1.e-4, 1.0 / 3.0 - 2.e-4, 0.0);
		// E
		Eigen::MatrixX<double> Dx0_xi_E = get_Dx0_xi_tmp(1.0 / 3.0 - 2.e-4, 1.0 / 3.0 + 1.e-4, 0.0);
		Eigen::MatrixX<autodiff::var> Dx_xi_E = get_Dx_xi_tmp(1.0 / 3.0 - 2.e-4, 1.0 / 3.0 + 1.e-4, 0.0);
		// F
		Eigen::MatrixX<double> Dx0_xi_F = get_Dx0_xi_tmp(1.0 / 3.0 + 1.e-4, 1.0 / 3.0 + 1.e-4, 0.0);
		Eigen::MatrixX<autodiff::var> Dx_xi_F = get_Dx_xi_tmp(1.0 / 3.0 + 1.e-4, 1.0 / 3.0 + 1.e-4, 0.0);



		double Dx0_xi_hat_rtst =
			Dx0_xi_F(0, 2) - Dx0_xi_D(0, 2) -
			Dx0_xi_F(1, 2) + Dx0_xi_E(1, 2);
		double Dx0_xi_rt = 2.0 / 3.0 * (Dx0_xi_B(0, 2) - 0.5 * Dx0_xi_B(1, 2)) +
			1.0 / 3.0 * (Dx0_xi_C(0, 2) + Dx0_xi_C(1, 2));// +Dx0_xi_hat_rtst * (xj - 1.0 / 3.0);
		double Dx0_xi_st = 2.0 / 3.0 * (Dx0_xi_A(1, 2) - 0.5 * Dx0_xi_A(0, 2)) +
			1.0 / 3.0 * (Dx0_xi_C(0, 2) + Dx0_xi_C(1, 2));// +Dx0_xi_hat_rtst * (1.0 / 3.0 - xi);

		double Dx0_xi_hat_trts =
			Dx0_xi_F(2, 0) - Dx0_xi_D(2, 0) -
			Dx0_xi_F(2, 1) + Dx0_xi_E(2, 1);
		double Dx0_xi_tr = 2.0 / 3.0 * (Dx0_xi_B(2, 0) - 0.5 * Dx0_xi_B(2, 1)) +
			1.0 / 3.0 * (Dx0_xi_C(2, 0) + Dx0_xi_C(2, 1)) + Dx0_xi_hat_trts * (xj - 1.0 / 3.0);
		double Dx0_xi_ts = 2.0 / 3.0 * (Dx0_xi_A(2, 1) - 0.5 * Dx0_xi_A(2, 0)) +
			1.0 / 3.0 * (Dx0_xi_C(2, 0) + Dx0_xi_C(2, 1)) + Dx0_xi_hat_trts * (1.0 / 3.0 - xi);

		//autodiff::var Dx_xi_hat_rtst =
		//	Dx_xi_F(0, 2) - Dx_xi_D(0, 2) -
		//	Dx_xi_F(1, 2) + Dx_xi_E(1, 2);
		//autodiff::var Dx_xi_rt = 2.0 / 3.0 * (Dx_xi_B(0, 2) - 0.5 * Dx_xi_B(1, 2)) +
		//	1.0 / 3.0 * (Dx_xi_C(0, 2) + Dx_xi_C(1, 2)) + Dx_xi_hat_rtst * (xj - 1.0 / 3.0);
		//autodiff::var Dx_xi_st = 2.0 / 3.0 * (Dx_xi_A(1, 2) - 0.5 * Dx_xi_A(0, 2)) +
		//	1.0 / 3.0 * (Dx_xi_C(0, 2) + Dx_xi_C(1, 2)) + Dx_xi_hat_rtst * (xj - 1.0 / 3.0);

		//autodiff::var Dx_xi_hat_trts =
		//	Dx_xi_F(2, 0) - Dx_xi_D(2, 0) -
		//	Dx_xi_F(2, 1) + Dx_xi_E(2, 1);
		//autodiff::var Dx_xi_tr = 2.0 / 3.0 * (Dx_xi_B(2, 0) - 0.5 * Dx_xi_B(2, 1)) +
		//	1.0 / 3.0 * (Dx_xi_C(2, 0) + Dx_xi_C(2, 1)) + Dx_xi_hat_trts * (xj - 1.0 / 3.0);
		//autodiff::var Dx_xi_ts = 2.0 / 3.0 * (Dx_xi_A(2, 1) - 0.5 * Dx_xi_A(2, 0)) +
		//	1.0 / 3.0 * (Dx_xi_C(2, 0) + Dx_xi_C(2, 1)) + Dx_xi_hat_trts * (xj - 1.0 / 3.0);



		Eigen::VectorXd N_xi = compute_shape_function_3x1(xi, xj);
		Eigen::MatrixXd DN_xi = compute_derivative_shape_function_2x3(xi, xj);

		Eigen::MatrixX<autodiff::var> Dx_xi = compute_derivative_position_3x3(xk, xlocal, thicklocal, Vnlocal_i, DN_xi, N_xi);
		Eigen::MatrixXd Dx0_xi = compute_derivative_position_3x3(xk, x0local, thicklocal, Vn0local, DN_xi, N_xi);
		DN_xi.conservativeResize(3, 3); // 3 x v_size


		std::cout <<
			Dx_xi(0, 2) << " " <<
			Dx_xi_A(0, 2) << " " <<
			Dx_xi_B(0, 2) << " " <<
			Dx_xi_C(0, 2) << " " <<
			Dx_xi_D(0, 2) << " " <<
			Dx_xi_E(0, 2) << " " <<
			Dx_xi_F(0, 2) << std::endl;
		std::cout <<
			Dx_xi(1, 2) << " " <<
			Dx_xi_A(1, 2) << " " <<
			Dx_xi_B(1, 2) << " " <<
			Dx_xi_C(1, 2) << " " <<
			Dx_xi_D(1, 2) << " " <<
			Dx_xi_E(1, 2) << " " <<
			Dx_xi_F(1, 2) << std::endl;
		std::cout <<
			Dx_xi(2, 0) << " " <<
			Dx_xi_A(2, 0) << " " <<
			Dx_xi_B(2, 0) << " " <<
			Dx_xi_C(2, 0) << " " <<
			Dx_xi_D(2, 0) << " " <<
			Dx_xi_E(2, 0) << " " <<
			Dx_xi_F(2, 0) << std::endl;
		std::cout <<
			Dx_xi(2, 1) << " " <<
			Dx_xi_A(2, 1) << " " <<
			Dx_xi_B(2, 1) << " " <<
			Dx_xi_C(2, 1) << " " <<
			Dx_xi_D(2, 1) << " " <<
			Dx_xi_E(2, 1) << " " <<
			Dx_xi_F(2, 1) << std::endl;


		// interpolation
		//std::cout << Dx0_xi.coeffRef(0, 2) << " " << 
		//	Dx0_xi_A(0, 2) << " " <<
		//	Dx0_xi_B(0, 2) << " " <<
		//	Dx0_xi_C(0, 2) << " " <<
		//	Dx0_xi_D(0, 2) << " " <<
		//	Dx0_xi_E(0, 2) << " " <<
		//	Dx0_xi_F(0, 2) << " " <<
		//	Dx0_xi_rt << std::endl;
		//std::cout << Dx0_xi.coeffRef(1, 2) << " " << Dx0_xi_st << std::endl;
		//std::cout << Dx0_xi.coeffRef(2, 0) << " " << Dx0_xi_tr << std::endl;
		//std::cout << Dx0_xi.coeffRef(2, 1) << " " << Dx0_xi_ts << std::endl;
		Dx0_xi.coeffRef(0, 2) = Dx0_xi_rt;
		Dx0_xi.coeffRef(1, 2) = Dx0_xi_st;
		Dx0_xi.coeffRef(2, 0) = Dx0_xi_tr;
		Dx0_xi.coeffRef(2, 1) = Dx0_xi_ts;

		//Dx_xi.coeffRef(0, 2) = Dx_xi_rt;
		//Dx_xi.coeffRef(1, 2) = Dx_xi_st;
		//Dx_xi.coeffRef(2, 0) = Dx_xi_tr;
		//Dx_xi.coeffRef(2, 1) = Dx_xi_ts;


		DN_xi.bottomRows(1).setZero();

		Eigen::Matrix3d Dxi_x0 = Dx0_xi.inverse(); // {xi/x0, xi/y0, xi/z0} {eta/x0, eta/y0, eta/z0} {zeta/x0, zeta/y0, zeta/z0}
		auto Dxi_x = Dx_xi.inverse(); // {xi/x, xi/y, xi/z} {eta/x, eta/y, eta/z} {zeta/x, zeta/y, zeta/z}

		Eigen::MatrixXd DN_x0 = Dxi_x0.transpose() * DN_xi; // 3 x v_size, {N1/x0, N2/x0, ...} {N1/y0, N2/y0, ...} {N1/z0, N2/z0, ...}
		auto DN_x = Dxi_x.transpose() * DN_xi; // 3 x v_size, {N1/x, N2/x, ...} {N1/y, N2/y, ...} {N1/z, N2/z, ...}


		//autodiff::var Jx_xi = abs(Dx_xi.determinant());
		Eigen::Matrix3<autodiff::var> F = xlocal * DN_x0.cast<autodiff::var>().transpose();
		//------ shell
		for (int anode = 0; anode < ivs_size; ++anode) {
			const auto& Na = N_xi[anode];
			F.col(0) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 0) * Na + xk * DN_x0(0, anode)) * Vnlocal_i.col(anode);
			F.col(1) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 1) * Na + xk * DN_x0(1, anode)) * Vnlocal_i.col(anode);
			F.col(2) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 2) * Na + xk * DN_x0(2, anode)) * Vnlocal_i.col(anode);
		}

		auto J = F.determinant();// max(1.e-12, F.determinant());
		////std::cout << F << std::endl;
		////std::cout << J << std::endl;
		//assert(J > 0.0);
		Eigen::Matrix3<autodiff::var> b = F * F.transpose();

		return 0.5 * mu * (b.trace() - 3.0) - mu * log(J) + 0.5 * lambda * log(J) * log(J);

	};


	autodiff::VectorXvar dofs2(5 * ivs_size);
	dofs2 = dofs;
	autodiff::var u = calc_phi2(dofs2);
	//auto ddofs = autodiff::gradient(u, dofs2);
	Eigen::VectorXd ddofs;
	auto dddofs = autodiff::hessian(u, dofs2, ddofs);

	std::cout << "val : " << u << std::endl;
	std::cout << ddofs << std::endl;
	std::cout << dddofs << std::endl;





	[&](Eigen::VectorXd dofs_in)->void {

		Eigen::MatrixXd Vnlocal_i(3, 3);
		Vnlocal_i.setZero();

		Eigen::MatrixXd V1local(3, 3);
		Eigen::MatrixXd V2local(3, 3);
		for (int anode = 0; anode < ivs_size; ++anode) {
			const auto& alpha = dofs_in(5 * anode + 3);
			const auto& beta = dofs_in(5 * anode + 4);
			Eigen::Vector3d Vn = Vnlocal.col(anode);
			auto tan_vec = get_tangent_vectors(Vn);
			Eigen::Vector3d tan0 = tan_vec.row(0);
			Eigen::Vector3d tan1 = tan_vec.row(1);

			V1local.col(anode) = tan0;
			V2local.col(anode) = tan1;

			// European Journal of Mechanics / A Solids 89 (2021) 104283
			Eigen::Vector3d theta = alpha * tan0 + beta * tan1;
			Eigen::Matrix3d Theta;
			Theta << 0.0, -theta[2], theta[1],
				theta[2], 0.0, -theta[0],
				-theta[1], theta[0], 0.0;

			Vnlocal_i.col(anode) =
				(Eigen::Matrix3d::Identity() + Theta) *
				Vnlocal.col(anode);

		}


		Eigen::MatrixXd xlocal(3, 3);
		xlocal.col(0) = dofs_in.segment(0, 3);
		xlocal.col(1) = dofs_in.segment(5, 3);
		xlocal.col(2) = dofs_in.segment(10, 3);


		//--------------------
		std::vector<Eigen::MatrixXd> B_a_rt(6);
		std::vector<Eigen::MatrixXd> B_a_st(6);
		Eigen::MatrixXd rs_tmp(2, 6);
		rs_tmp.col(0) << 1.0 / 6.0, 2.0 / 3.0;
		rs_tmp.col(1) << 2.0 / 3.0, 1.0 / 6.0;
		rs_tmp.col(2) << 1.0 / 6.0, 1.0 / 6.0;
		rs_tmp.col(3) << 1.0 / 3.0 + 1.e-4, 1.0 / 3.0 - 2.e-4;
		rs_tmp.col(4) << 1.0 / 3.0 - 2.e-4, 1.0 / 3.0 + 1.e-4;
		rs_tmp.col(5) << 1.0 / 3.0 + 1.e-4, 1.0 / 3.0 + 1.e-4;
		for (int irs = 0; irs < 6; ++irs)
		{
			Eigen::VectorXd N_xi = compute_shape_function_3x1(rs_tmp(0, irs), rs_tmp(1, irs));
			Eigen::MatrixXd DN_xi = compute_derivative_shape_function_2x3(rs_tmp(0, irs), rs_tmp(1, irs));
			Eigen::MatrixXd Dx_xi = compute_derivative_position_3x3(0.0, xlocal, thicklocal, Vnlocal_i, DN_xi, N_xi);
			auto Dxi_x = Dx_xi.inverse();
			DN_xi.conservativeResize(3, 3);
			DN_xi.bottomRows(1).setZero();
			auto DN_x = Dxi_x.transpose() * DN_xi;

			auto& B_a_rt_i = B_a_rt[irs];
			auto& B_a_st_i = B_a_st[irs];
			B_a_rt_i.resize(3, 5);
			B_a_st_i.resize(3, 5);
			for (int anode = 0; anode < ivs_size; ++anode) {
				const auto& Na = N_xi[anode];
				const auto& thick = thicklocal[anode];
				B_a_st_i.row(anode) << 0.0, DN_x(2, anode), DN_x(1, anode),
					-0.5 * thick * (
						(Dxi_x(2, 1) * Na) * V2local(2, anode) +
						(Dxi_x(2, 2) * Na) * V2local(1, anode)),
					0.5 * thick * (
						(Dxi_x(2, 1) * Na) * V1local(2, anode) +
						(Dxi_x(2, 2) * Na) * V1local(1, anode));

				B_a_rt_i.row(anode) << DN_x(2, anode), 0.0, DN_x(0, anode),
					-0.5 * thick * (
						(Dxi_x(2, 0) * Na) * V2local(2, anode) +
						(Dxi_x(2, 2) * Na) * V2local(0, anode)),
					0.5 * thick * (
						(Dxi_x(2, 0) * Na) * V1local(2, anode) +
						(Dxi_x(2, 2) * Na) * V1local(0, anode));
			}
		}
		Eigen::MatrixXd B_hat_rtst = B_a_rt[5] - B_a_rt[3] - B_a_st[5] + B_a_st[4];
		Eigen::MatrixXd B_bar_rt = 2.0 / 3.0 * (B_a_rt[1] - 0.5 * B_a_st[1]) +
			1.0 / 3.0 * (B_a_rt[2] + B_a_st[2]) + B_hat_rtst * (xj - 1.0 / 3.0);
		Eigen::MatrixXd B_bar_st = 2.0 / 3.0 * (B_a_st[0] - 0.5 * B_a_rt[0]) +
			1.0 / 3.0 * (B_a_rt[2] + B_a_st[2]) + B_hat_rtst * (1.0 / 3.0 - xi);

		//--------------------



		Eigen::VectorXd N_xi = compute_shape_function_3x1(xi, xj);
		Eigen::MatrixXd DN_xi = compute_derivative_shape_function_2x3(xi, xj);

		Eigen::MatrixXd Dx_xi = compute_derivative_position_3x3(xk, xlocal, thicklocal, Vnlocal_i, DN_xi, N_xi);
		Eigen::MatrixXd Dx0_xi = compute_derivative_position_3x3(xk, x0local, thicklocal, Vn0local, DN_xi, N_xi);
		DN_xi.conservativeResize(3, 3); // 3 x v_size

		DN_xi.bottomRows(1).setZero();

		Eigen::Matrix3d Dxi_x0 = Dx0_xi.inverse(); // {xi/x0, xi/y0, xi/z0} {eta/x0, eta/y0, eta/z0} {zeta/x0, zeta/y0, zeta/z0}
		auto Dxi_x = Dx_xi.inverse(); // {xi/x, xi/y, xi/z} {eta/x, eta/y, eta/z} {zeta/x, zeta/y, zeta/z}

		Eigen::MatrixXd DN_x0 = Dxi_x0.transpose() * DN_xi; // 3 x v_size, {N1/x0, N2/x0, ...} {N1/y0, N2/y0, ...} {N1/z0, N2/z0, ...}
		auto DN_x = Dxi_x.transpose() * DN_xi; // 3 x v_size, {N1/x, N2/x, ...} {N1/y, N2/y, ...} {N1/z, N2/z, ...}


		//autodiff::var Jx_xi = abs(Dx_xi.determinant());
		Eigen::Matrix3d F = xlocal * DN_x0.transpose();
		//------ shell
		for (int anode = 0; anode < ivs_size; ++anode) {
			const auto& Na = N_xi[anode];
			F.col(0) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 0) * Na + xk * DN_x0(0, anode)) * Vnlocal_i.col(anode);
			F.col(1) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 1) * Na + xk * DN_x0(1, anode)) * Vnlocal_i.col(anode);
			F.col(2) += 0.5 * thicklocal[anode] * (Dxi_x0(2, 2) * Na + xk * DN_x0(2, anode)) * Vnlocal_i.col(anode);
		}

		auto J = F.determinant();// max(1.e-12, F.determinant());
		////std::cout << F << std::endl;
		////std::cout << J << std::endl;
		//assert(J > 0.0);
		Eigen::Matrix3d b = F * F.transpose();

		//return 0.5 * mu * (b.trace() - 3.0) - mu * log(J) + 0.5 * lambda * log(J) * log(J);





		Eigen::Matrix3d stress;
		stress = cauchy_stress(mu, lambda, J, b);

		Eigen::VectorXd stress_Voigt(6);
		stress_Voigt << stress(0, 0), stress(1, 1), stress(2, 2),
			stress(1, 2), stress(0, 2), stress(0, 1);


		std::vector<Eigen::MatrixXd> B_as(ivs_size);
		for (int anode = 0; anode < ivs_size; ++anode) {

			const auto& Na = N_xi[anode];
			const auto& thick = thicklocal[anode];

			Eigen::MatrixXd& B_a = B_as[anode];
			B_a.resize(6, 5);

			B_a.row(0) << DN_x(0, anode), 0.0, 0.0,
				-0.5 * thick * (Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) * V2local(0, anode),
				0.5 * thick * (Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) * V1local(0, anode);
			B_a.row(1) << 0.0, DN_x(1, anode), 0.0,
				-0.5 * thick * (Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) * V2local(1, anode),
				0.5 * thick * (Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) * V1local(1, anode);
			B_a.row(2) << 0.0, 0.0, DN_x(2, anode),
				-0.5 * thick * (Dxi_x(2, 2) * Na + xk * DN_x(2, anode)) * V2local(2, anode),
				0.5 * thick * (Dxi_x(2, 2) * Na + xk * DN_x(2, anode)) * V1local(2, anode);

			B_a.row(3) << 0.0, DN_x(2, anode), DN_x(1, anode),
				-0.5 * thick * (
					(Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) * V2local(2, anode) +
					(Dxi_x(2, 2) * Na + xk * DN_x(2, anode)) * V2local(1, anode)),
				0.5 * thick * (
					(Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) * V1local(2, anode) +
					(Dxi_x(2, 2) * Na + xk * DN_x(2, anode)) * V1local(1, anode));
			B_a.row(4) << DN_x(2, anode), 0.0, DN_x(0, anode),
				-0.5 * thick * (
					(Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) * V2local(2, anode) +
					(Dxi_x(2, 2) * Na + xk * DN_x(2, anode)) * V2local(0, anode)),
				0.5 * thick * (
					(Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) * V1local(2, anode) +
					(Dxi_x(2, 2) * Na + xk * DN_x(2, anode)) * V1local(0, anode));

			//B_a.row(3) = B_bar_st.row(anode);
			//B_a.row(4) = B_bar_rt.row(anode);

			B_a.row(5) << DN_x(1, anode), DN_x(0, anode), 0.0,
				-0.5 * thick * (
					(Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) * V2local(1, anode) +
					(Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) * V2local(0, anode)),
				0.5 * thick * (
					(Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) * V1local(1, anode) +
					(Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) * V1local(0, anode));
		}

		Eigen::VectorXd gradient(5 * ivs_size);
		for (int anode = 0; anode < ivs_size; ++anode) {

			const auto& Na = N_xi[anode];
			const auto& thick = thicklocal[anode];

			Eigen::MatrixXd& B_a = B_as[anode];

			gradient.segment(anode * 5, 5) = B_a.transpose() * stress_Voigt;

		}

		std::cout << gradient * J << std::endl;

		//Eigen::Vector3d V1 = V1local.col(0);
		//Eigen::Vector3d V2 = V2local.col(0);
		//Eigen::Vector3d Vn = Vnlocal.col(0);
		//std::cout << V1 << std::endl << std::endl;
		//std::cout << V2 << std::endl << std::endl;
		//std::cout << V1.cross(Vn) << std::endl << std::endl;
		//std::cout << V2.cross(Vn) << std::endl << std::endl;




		Eigen::MatrixXd modulus_Voigt = elasticity_modulus_Voigt(mu, lambda, J);


		Eigen::MatrixXd hess(5 * ivs_size, 5 * ivs_size);

		//Eigen::MatrixXd DN_sigma_DN = DN_x.transpose() * grad;
		for (int anode = 0; anode < ivs_size; ++anode) {
			const Eigen::MatrixXd& B_a = B_as[anode];
			const auto& Na = N_xi[anode];
			const auto& thick_a = thicklocal[anode];
			for (int bnode = 0; bnode < ivs_size; ++bnode) {
				const Eigen::MatrixXd& B_b = B_as[bnode];
				const auto& Nb = N_xi[bnode];
				const auto& thick_b = thicklocal[bnode];

				//---------------------
				// constitutive component
				hess.block(anode * 5, bnode * 5, 5, 5) =
					B_a.transpose() * modulus_Voigt * B_b;


				//---------------------
				// initial stress component
				double coeff1 = (
					stress(0, 0) * DN_x(0, anode) +
					stress(1, 0) * DN_x(1, anode) +
					stress(2, 0) * DN_x(2, anode));
				double coeff2 = (
					stress(0, 1) * DN_x(0, anode) +
					stress(1, 1) * DN_x(1, anode) +
					stress(2, 1) * DN_x(2, anode));
				double coeff3 = (
					stress(0, 2) * DN_x(0, anode) +
					stress(1, 2) * DN_x(1, anode) +
					stress(2, 2) * DN_x(2, anode));

				double A_11_comp =  
					DN_x(0, bnode) * coeff1 +
					DN_x(1, bnode) * coeff2 +
					DN_x(2, bnode) * coeff3;
				hess.block(anode * 5, bnode * 5, 3, 3) +=
					A_11_comp * Eigen::Matrix3d::Identity();
				
				double A_12_comp =
					-0.5 * thick_b * (Dxi_x(2, 0) * Nb + xk * DN_x(0, bnode)) * coeff1
					-0.5 * thick_b * (Dxi_x(2, 1) * Nb + xk * DN_x(1, bnode)) * coeff2
					-0.5 * thick_b * (Dxi_x(2, 2) * Nb + xk * DN_x(2, bnode)) * coeff3;
				hess.block(anode * 5, bnode * 5 + 3, 3, 1) +=
					A_12_comp * V2local.col(bnode);

				double A_13_comp = -A_12_comp;
				hess.block(anode * 5, bnode * 5 + 4, 3, 1) +=
					A_13_comp * V1local.col(bnode);

				double coeff21 = (
					stress(0, 0) * (Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) +
					stress(1, 0) * (Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) +
					stress(2, 0) * (Dxi_x(2, 2) * Na + xk * DN_x(2, anode)));
				double coeff22 = (
					stress(0, 1) * (Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) +
					stress(1, 1) * (Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) +
					stress(2, 1) * (Dxi_x(2, 2) * Na + xk * DN_x(2, anode)));
				double coeff23 = (
					stress(0, 2) * (Dxi_x(2, 0) * Na + xk * DN_x(0, anode)) +
					stress(1, 2) * (Dxi_x(2, 1) * Na + xk * DN_x(1, anode)) +
					stress(2, 2) * (Dxi_x(2, 2) * Na + xk * DN_x(2, anode)));

				double A_21_comp =
					-0.5 * thick_a * DN_x(0, bnode) * coeff21
					- 0.5 * thick_a * DN_x(1, bnode) * coeff22
					- 0.5 * thick_a * DN_x(2, bnode) * coeff23;
				hess.block(anode * 5 + 3, bnode * 5, 1, 3) +=
					A_21_comp * V2local.col(anode).transpose();

				double A_22_comp =
					+0.25 * thick_a * thick_b * (Dxi_x(2, 0) * Nb + xk * DN_x(0, bnode)) * coeff21
					+0.25 * thick_a * thick_b * (Dxi_x(2, 1) * Nb + xk * DN_x(1, bnode)) * coeff22
					+0.25 * thick_a * thick_b * (Dxi_x(2, 2) * Nb + xk * DN_x(2, bnode)) * coeff23;
				hess.coeffRef(anode * 5 + 3, bnode * 5 + 3) +=
					A_22_comp * V2local.col(anode).dot(V2local.col(bnode));

				double A_23_comp = -A_22_comp;
				hess.coeffRef(anode * 5 + 3, bnode * 5 + 4) +=
					A_23_comp * V2local.col(anode).dot(V1local.col(bnode));

				//----------

				double A_31_comp = -A_21_comp;
				hess.block(anode * 5 + 4, bnode * 5, 1, 3) +=
					A_31_comp * V1local.col(anode).transpose();

				double A_32_comp = A_23_comp;
				hess.coeffRef(anode * 5 + 4, bnode * 5 + 3) +=
					A_32_comp * V1local.col(anode).dot(V2local.col(bnode));

				double A_33_comp = A_22_comp;
				hess.coeffRef(anode * 5 + 4, bnode * 5 + 4) +=
					A_33_comp * V1local.col(anode).dot(V1local.col(bnode));
			}
		}

		std::cout << hess * J << std::endl;


	}(dofs);




}