
#include <iostream>
#include <random>

#include <Eigen/Dense>
#include <finite-diff/finitediff.hpp>
#include "./3rd-party/autodiff/reverse/var.hpp"
#include "./3rd-party/autodiff/reverse/var/eigen.hpp"

class GaussQuadrature {
public:

	class Quadrature {
	public:
		int size{};
		std::array<double, 36> xi{};
		std::array<double, 12> w{};

		//Quadrature(
		//    const int& N_in,
		//    const std::array<double, 12>& xi_in,
		//    const std::array<double, 12>& w_in) :
		//    size(N_in), xi(xi_in), w(w_in)
		//{
		//}

		Quadrature() = default;

		Quadrature(
			const std::vector<double>& xi_in,
			const std::vector<double>& w_in
		)
		{
			size = w_in.size();
			assert(xi_in.size() / 3 == w_in.size());
			for (int i = 0; auto & item : xi_in) {
				xi[i++] = item;
			}
			for (int i = 0; auto & item : w_in) {
				w[i++] = item;
			}
		}
	};

	// edge
	inline static auto edge2 = Quadrature(
		{
			-0.577350269189626, 0.0, 0.0,
			0.577350269189626, 0.0, 0.0 },
		{ 1.0, 1.0 }
		);
	inline static auto edge3 = Quadrature(
		{
			0.774596669241483, 0.0, 0.0,
			0.0, 0.0, 0.0,
			-0.774596669241483, 0.0, 0.0 },
		{ 0.555555555555554, 0.888888888888889, 0.555555555555554 }
		);
	// triangle
	inline static auto tria1 = Quadrature(
		{ 1.0 / 3.0, 1.0 / 3.0, 0.0 },
		{ 1.0 }
	);
	inline static auto tria3 = Quadrature(
		//{ 0.5, 0.0, 0.0, 
		//  0.5, 0.5, 0.0, 
		//  0.0, 0.5, 0.0 },
		{ 2.0 / 3.0, 1.0 / 6.0, 0.0,
		  1.0 / 6.0, 2.0 / 3.0, 0.0,
		  1.0 / 6.0, 1.0 / 6.0, 0.0 },
		{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 }
	);
	inline static auto tria6 = Quadrature(
		{
			 4.459484909159650e-01, 4.459484909159650e-01, 0.0 ,
			 4.459484909159650e-01, 1.081030181680700e-01, 0.0 ,
			 1.081030181680700e-01, 4.459484909159650e-01, 0.0 ,
			 9.157621350977102e-02, 9.157621350977102e-02, 0.0 ,
			 9.157621350977102e-02, 8.168475729804590e-01, 0.0 ,
			 8.168475729804590e-01, 9.157621350977102e-02, 0.0
		},
			{
				0.2233815897, 0.2233815897,
				0.2233815897, 0.1099517437,
				0.1099517437, 0.1099517437
			}
	);
	// quadratic
	inline static auto quad4 = Quadrature(
		{
			-0.577350269189626, -0.577350269189626, 0.0 ,
			 0.577350269189626, -0.577350269189626, 0.0 ,
			 0.577350269189626,  0.577350269189626, 0.0 ,
			-0.577350269189626,  0.577350269189626, 0.0
		},
		{ 1.0, 1.0, 1.0, 1.0 }
	);
	// prism
	inline static auto prism2 = Quadrature(
		{ 1.0 / 3.0, 1.0 / 3.0, -0.577350269189626,
		  1.0 / 3.0, 1.0 / 3.0,  0.577350269189626 },
		{ 0.5, 0.5 }
	);
	inline static auto prism6 = Quadrature(
		{ 1.0 / 6.0, 1.0 / 6.0, -0.577350269189626,
		  2.0 / 3.0, 1.0 / 6.0, -0.577350269189626,
		  1.0 / 6.0, 2.0 / 3.0, -0.577350269189626,
		  1.0 / 6.0, 1.0 / 6.0,  0.577350269189626,
		  2.0 / 3.0, 1.0 / 6.0,  0.577350269189626,
		  1.0 / 6.0, 2.0 / 3.0,  0.577350269189626 },
		{ 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,
		  1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 }
	);
	inline static auto prism8 = Quadrature(
		{ 1.0 / 3.0, 1.0 / 3.0, -0.577350269189626,
		  1.0 / 5.0, 1.0 / 5.0, -0.577350269189626,
		  3.0 / 5.0, 1.0 / 5.0, -0.577350269189626,
		  1.0 / 5.0, 3.0 / 5.0, -0.577350269189626,
		  1.0 / 3.0, 1.0 / 3.0,  0.577350269189626,
		  1.0 / 5.0, 1.0 / 5.0,  0.577350269189626,
		  3.0 / 5.0, 1.0 / 5.0,  0.577350269189626,
		  1.0 / 5.0, 3.0 / 5.0,  0.577350269189626 },
		{ -9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0,
		  -9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0 }
	);
	inline static auto prism12 = Quadrature(
		{
			 4.459484909159650e-01, 4.459484909159650e-01, -0.577350269189626 ,
			 4.459484909159650e-01, 1.081030181680700e-01, -0.577350269189626 ,
			 1.081030181680700e-01, 4.459484909159650e-01, -0.577350269189626 ,
			 9.157621350977102e-02, 9.157621350977102e-02, -0.577350269189626 ,
			 9.157621350977102e-02, 8.168475729804590e-01, -0.577350269189626 ,
			 8.168475729804590e-01, 9.157621350977102e-02, -0.577350269189626 ,
			 4.459484909159650e-01, 4.459484909159650e-01, 0.577350269189626 ,
			 4.459484909159650e-01, 1.081030181680700e-01, 0.577350269189626 ,
			 1.081030181680700e-01, 4.459484909159650e-01, 0.577350269189626 ,
			 9.157621350977102e-02, 9.157621350977102e-02, 0.577350269189626 ,
			 9.157621350977102e-02, 8.168475729804590e-01, 0.577350269189626 ,
			 8.168475729804590e-01, 9.157621350977102e-02, 0.577350269189626
		},
			{
				0.1116907948390055, 0.1116907948390055,
				0.1116907948390055, 0.054975871827661,
				0.054975871827661, 0.054975871827661,
				0.1116907948390055, 0.1116907948390055,
				0.1116907948390055, 0.054975871827661,
				0.054975871827661, 0.054975871827661
			}
	);
	// tetrahedron
	inline static auto tetr1 = Quadrature(
		{
			0.25,0.25,0.25
		},
		{ 0.16666666666666666 }
	);
	inline static auto tetr4 = Quadrature(
		{
			0.5854102, 0.1381966, 0.1381966,
			0.1381966, 0.5854102, 0.1381966,
			0.1381966, 0.1381966, 0.5854102,
			0.1381966, 0.1381966, 0.1381966
		},
		{ 0.0416667, 0.0416667, 0.0416667, 0.0416667 }
	);
	inline static auto tetr5 = Quadrature(
		{
			0.25,0.25,0.25,
			1.0 / 6.0,1.0 / 6.0,1.0 / 6.0,
			0.5,1.0 / 6.0,1.0 / 6.0,
			1.0 / 6.0,0.5,1.0 / 6.0,
			1.0 / 6.0,1.0 / 6.0,0.5
		},
		{ -0.1333333333333333, 0.075, 0.075, 0.075, 0.075 }
	);
	inline static auto tetr11 = Quadrature(
		{
			 2.500000000000000e-01, 2.500000000000000e-01, 2.500000000000000e-01 ,
			 7.142857142857151e-02, 7.142857142857151e-02, 7.142857142857151e-02 ,
			 7.142857142857151e-02, 7.142857142857151e-02, 7.857142857142855e-01 ,
			 7.142857142857151e-02, 7.857142857142855e-01, 7.142857142857151e-02 ,
			 7.857142857142855e-01, 7.142857142857151e-02, 7.142857142857151e-02 ,
			 3.994035761667990e-01, 3.994035761667990e-01, 1.005964238332010e-01 ,
			 3.994035761667990e-01, 1.005964238332010e-01, 3.994035761667990e-01 ,
			 1.005964238332010e-01, 3.994035761667990e-01, 3.994035761667990e-01 ,
			 3.994035761667990e-01, 1.005964238332010e-01, 1.005964238332010e-01 ,
			 1.005964238332010e-01, 3.994035761667990e-01, 1.005964238332010e-01 ,
			 1.005964238332010e-01, 1.005964238332010e-01, 3.994035761667990e-01
		},
			{
				-1.315555555555550e-02, 7.622222222222250e-03,
				7.622222222222250e-03, 7.622222222222250e-03,
				7.622222222222250e-03, 2.488888888888887e-02,
				2.488888888888887e-02, 2.488888888888887e-02,
				2.488888888888887e-02, 2.488888888888887e-02,
				2.488888888888887e-02
			}
	);
	// hexahedron
	inline static auto hexa1 = Quadrature(
		{
			0.0, 0.0, 0.0
		},
		{ 8.0 }
	);
	inline static auto hexa2 = Quadrature(
		{
			0.0, 0.0, -0.577350269189626,
			0.0, 0.0,  0.577350269189626
		},
		{ 4.0, 4.0 }
	);
	inline static auto hexa8 = Quadrature(
		{
			-0.577350269189626, -0.577350269189626, -0.577350269189626,
			 0.577350269189626, -0.577350269189626, -0.577350269189626,
			 0.577350269189626,  0.577350269189626, -0.577350269189626,
			-0.577350269189626,  0.577350269189626, -0.577350269189626,
			-0.577350269189626, -0.577350269189626,  0.577350269189626,
			 0.577350269189626, -0.577350269189626,  0.577350269189626,
			 0.577350269189626,  0.577350269189626,  0.577350269189626,
			-0.577350269189626,  0.577350269189626,  0.577350269189626
		},
		{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
	);
	// pyramidal
	inline static auto pyramid5 = Quadrature(
		{
			0.0,  0.0, 0.25,
			-0.5,  0.0, 0.25,
			0.5,  0.0, 0.25,
			0.0, -0.5, 0.25,
			0.0,  0.5, 0.25
		},
			{
				-0.8,
				0.45, 0.45, 0.45, 0.45
			}
	);
};


namespace MITC4 {


	inline Eigen::VectorXd compute_shape_function_4x1(
		const double& r, const double& s
	) {
		return (Eigen::VectorXd(4) <<
			0.25 * (1.0 - r) * (1.0 - s),
			0.25 * (1.0 + r) * (1.0 - s),
			0.25 * (1.0 + r) * (1.0 + s),
			0.25 * (1.0 - r) * (1.0 + s)).finished();
	}

	inline Eigen::MatrixXd compute_derivative_shape_function_2x4(
		const double& r, const double& s
	) {
		return (Eigen::MatrixXd(2, 4) <<
			-0.25 * (1.0 - s),
			0.25 * (1.0 - s),
			0.25 * (1.0 + s),
			-0.25 * (1.0 + s),
			-0.25 * (1.0 - r),
			-0.25 * (1.0 + r),
			0.25 * (1.0 + r),
			0.25 * (1.0 - r)).finished();
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
		Eigen::MatrixX<T> DN_xi_var = DN_xi.transpose();
		Dx_xi.block(0, 0, 3, 2) = xlocal * DN_xi_var;
		for (int i = 0; i < 4; ++i) {
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



	template<typename T>
	Eigen::MatrixX<T> compute_derivative_shape_displacement_20x9(
		const double& t,
		const Eigen::MatrixX<T>& xlocal,
		const Eigen::VectorXd& thicklocal,
		const Eigen::MatrixXd& V1local,
		const Eigen::MatrixXd& V2local,
		const Eigen::MatrixX<T>& Vnlocal,
		const Eigen::MatrixXd& DN_xi,
		const Eigen::VectorXd& N_xi
	) {

		Eigen::MatrixX<T> dNdR(20, 9);

		for (int xidim = 0; xidim < 2; ++xidim) {
			for (int idof = 0; idof < 4; ++idof) {
				dNdR(5 * idof + 0, xidim * 3 + 0) = DN_xi(xidim, idof);
				dNdR(5 * idof + 1, xidim * 3 + 0) = 0;
				dNdR(5 * idof + 2, xidim * 3 + 0) = 0;

				dNdR(5 * idof + 0, xidim * 3 + 1) = 0;
				dNdR(5 * idof + 1, xidim * 3 + 1) = DN_xi(xidim, idof);
				dNdR(5 * idof + 2, xidim * 3 + 1) = 0;

				dNdR(5 * idof + 0, xidim * 3 + 2) = 0;
				dNdR(5 * idof + 1, xidim * 3 + 2) = 0;
				dNdR(5 * idof + 2, xidim * 3 + 2) = DN_xi(xidim, idof);
			}
		}

		for (int xidim = 0; xidim < 2; ++xidim) {
			for (int idof = 0; idof < 4; ++idof) {
				for (int idim = 0; idim < 3; ++idim) {
					dNdR(5 * idof + 3, xidim * 3 + idim) = -0.5 * t * thicklocal[idof] * DN_xi(xidim, idof) * V2local(idim, idof);
					dNdR(5 * idof + 4, xidim * 3 + idim) = 0.5 * t * thicklocal[idof] * DN_xi(xidim, idof) * V1local(idim, idof);
				}
			}
		}

		for (int idim = 0; idim < 3; ++idim) {
			dNdR.col(6 + idim) <<
				0, 0, 0, -0.5 * thicklocal[0] * N_xi[0] * V2local(idim, 0), 0.5 * thicklocal[0] * N_xi[0] * V1local(idim, 0),
				0, 0, 0, -0.5 * thicklocal[1] * N_xi[1] * V2local(idim, 1), 0.5 * thicklocal[1] * N_xi[1] * V1local(idim, 1),
				0, 0, 0, -0.5 * thicklocal[2] * N_xi[2] * V2local(idim, 2), 0.5 * thicklocal[2] * N_xi[2] * V1local(idim, 2),
				0, 0, 0, -0.5 * thicklocal[3] * N_xi[3] * V2local(idim, 3), 0.5 * thicklocal[3] * N_xi[3] * V1local(idim, 3);
		}

		return dNdR;
	}





	Eigen::MatrixXd compute_derivative_shape_displacement_9x5(
		const double& t,
		int idof,
		const Eigen::MatrixXd& xlocal,
		const Eigen::VectorXd& thicklocal,
		const Eigen::MatrixXd& V1local,
		const Eigen::MatrixXd& V2local,
		const Eigen::MatrixXd& Vnlocal,
		const Eigen::MatrixXd& DN_xi,
		const Eigen::VectorXd& N_xi
	) {

		Eigen::MatrixXd dNdR(9, 5);
		for (int xidim = 0; xidim < 2; ++xidim) {
			dNdR.block<3, 3>(3 * xidim, 0) = DN_xi(xidim, idof) * Eigen::Matrix3d::Identity();
		}

		for (int xidim = 0; xidim < 2; ++xidim) {
			for (int idim = 0; idim < 3; ++idim) {
				dNdR(xidim * 3 + idim, 3) = -0.5 * t * thicklocal[idof] * DN_xi(xidim, idof) * V2local(idim, idof);
				dNdR(xidim * 3 + idim, 4) = 0.5 * t * thicklocal[idof] * DN_xi(xidim, idof) * V1local(idim, idof);
			}
		}

		for (int idim = 0; idim < 3; ++idim) {
			dNdR.row(6 + idim) <<
				0, 0, 0,
				-0.5 * thicklocal[idof] * N_xi[idof] * V2local(idim, idof),
				0.5 * thicklocal[idof] * N_xi[idof] * V1local(idim, idof);
		}

		return dNdR;
	}


}


namespace Soft {


	inline Eigen::Matrix3d compute_contravariant(
		const Eigen::Vector3d& g1, const Eigen::Vector3d& g2, const Eigen::Vector3d& g3
	) {
		Eigen::Matrix3d gg;
		gg <<
			g1.dot(g1), g1.dot(g2), g1.dot(g3),
			g2.dot(g1), g2.dot(g2), g2.dot(g3),
			g3.dot(g1), g3.dot(g2), g3.dot(g3);
		Eigen::Matrix3d cgij = gg.fullPivLu().inverse();

		Eigen::Matrix3d cg;
		cg.col(0) = cgij(0, 0) * g1 + cgij(0, 1) * g2 + cgij(0, 2) * g3;
		cg.col(1) = cgij(1, 0) * g1 + cgij(1, 1) * g2 + cgij(1, 2) * g3;
		cg.col(2) = cgij(2, 0) * g1 + cgij(2, 1) * g2 + cgij(2, 2) * g3;

		return cg;

	}

	inline Eigen::Matrix3d compute_local_basis(
		const Eigen::Vector3d& g1, const Eigen::Vector3d& g2, const Eigen::Vector3d& g3
	) {

		Eigen::Matrix3d ehat;

		ehat.col(2) = g3.normalized();
		Eigen::Vector3d temp_col0 = g2.cross(ehat.col(2));
		if (temp_col0.norm() < 1.e-8) { // 또는 더 큰 임계값
			// g2와 g3가 거의 평행함
			// 대안적인 방법으로 ehat.col(0) 생성
			// 예를 들어, g3에 수직인 임의의 벡터를 찾아서 사용
			// (g3가 z축이 아니라면, (0,0,1)과 g3의 외적을 시도하고,
			//  그마저도 평행하다면 (0,1,0)과 g3의 외적을 시도하는 등)
			if (ehat.col(2).cross(Eigen::Vector3d::UnitX()).norm() > 1.e-8) {
				ehat.col(0) = ehat.col(2).cross(Eigen::Vector3d::UnitX()).normalized();
			}
			else {
				ehat.col(0) = ehat.col(2).cross(Eigen::Vector3d::UnitY()).normalized();
			}
		}
		else {
			ehat.col(0) = temp_col0.normalized();
		}
		ehat.col(1) = ehat.col(2).cross(ehat.col(0)).normalized();

		return ehat;

	}

	inline Eigen::MatrixXd compute_transform_matrix(
		const Eigen::Vector3d& s1, const Eigen::Vector3d& s2, const Eigen::Vector3d& s3,
		const Eigen::Vector3d& t1, const Eigen::Vector3d& t2, const Eigen::Vector3d& t3
	) {
		// finite element procedures, Klaus-Jürgen Bathe
		// eq 5.120
		double l1 = t1.dot(s1); double m1 = t2.dot(s1); double n1 = t3.dot(s1);
		double l2 = t1.dot(s2); double m2 = t2.dot(s2); double n2 = t3.dot(s2);
		double l3 = t1.dot(s3); double m3 = t2.dot(s3); double n3 = t3.dot(s3);

		Eigen::MatrixXd Q(6, 6);
		Q <<
			l1 * l1, m1* m1, n1* n1, m1* n1, l1* n1, l1* m1,
			l2* l2, m2* m2, n2* n2, m2* n2, l2* n2, l2* m2,
			l3* l3, m3* m3, n3* n3, m3* n3, l3* n3, l3* m3,
			2.0 * l2 * l3, 2.0 * m2 * m3, 2.0 * n2 * n3, (m2 * n3 + m3 * n2), (l2 * n3 + l3 * n2), (l2 * m3 + l3 * m2),
			2.0 * l1 * l3, 2.0 * m1 * m3, 2.0 * n1 * n3, (m1 * n3 + m3 * n1), (l1 * n3 + l3 * n1), (l1 * m3 + l3 * m1),
			2.0 * l1 * l2, 2.0 * m1 * m2, 2.0 * n1 * n2, (m1 * n2 + m2 * n1), (l1 * n2 + l2 * n1), (l1 * m2 + l2 * m1);


		//double l1 = t1.dot(s1); double m1 = t1.dot(s2); double n1 = t1.dot(s3);
		//double l2 = t2.dot(s1); double m2 = t2.dot(s2); double n2 = t2.dot(s3);
		//double l3 = t3.dot(s1); double m3 = t3.dot(s2); double n3 = t3.dot(s3);

		//Eigen::MatrixXd Q(6, 6);
		////Q <<
		////	l1 * l1, m1* m1, n1* n1, m1* n1, l1* n1, l1* m1,
		////	l2* l2, m2* m2, n2* n2, m2* n2, l2* n2, l2* m2,
		////	l3* l3, m3* m3, n3* n3, m3* n3, l3* n3, l3* m3,
		////	2.0 * l2 * l3, 2.0 * m2 * m3, 2.0 * n2 * n3, (m2 * n3 + m3 * n2), (l2 * n3 + l3 * n2), (l2 * m3 + l3 * m2),
		////	2.0 * l1 * l3, 2.0 * m1 * m3, 2.0 * n1 * n3, (m1 * n3 + m3 * n1), (l1 * n3 + l3 * n1), (l1 * m3 + l3 * m1),
		////	2.0 * l1 * l2, 2.0 * m1 * m2, 2.0 * n1 * n2, (m1 * n2 + m2 * n1), (l1 * n2 + l2 * n1), (l1 * m2 + l2 * m1);

		//Q <<
		//	l1 * l1, m1* m1, n1* n1, 2.0 * m1 * n1, 2.0 * l1 * n1, 2.0 * l1 * m1,
		//	l2* l2, m2* m2, n2* n2, 2.0 * m2 * n2, 2.0 * l2 * n2, 2.0 * l2 * m2,
		//	l3* l3, m3* m3, n3* n3, 2.0 * m3 * n3, 2.0 * l3 * n3, 2.0 * l3 * m3,
		//	l2* l3, m2* m3, n2* n3, (m2 * n3 + m3 * n2), (l2 * n3 + l3 * n2), (l2 * m3 + l3 * m2),
		//	l1* l3, m1* m3, n1* n3, (m1 * n3 + m3 * n1), (l1 * n3 + l3 * n1), (l1 * m3 + l3 * m1),
		//	l1* l2, m1* m2, n1* n2, (m1 * n2 + m2 * n1), (l1 * n2 + l2 * n1), (l1 * m2 + l2 * m1);


		return Q;

	}


	inline Eigen::VectorXd tensor_to_Voigt(
		const Eigen::Matrix3d& tensor
	) {

		Eigen::VectorXd tensor_Voigt(6);
		tensor_Voigt <<
			tensor(0, 0),
			tensor(1, 1),
			tensor(2, 2),
			2.0 * tensor(1, 2),
			2.0 * tensor(0, 2),
			2.0 * tensor(0, 1);
		return tensor_Voigt;
	}


	inline Eigen::Matrix3d Voigt_to_tensor(
		const Eigen::VectorXd& tensor_Voigt
	) {
		Eigen::Matrix3d tensor;
		tensor <<
			tensor_Voigt(0), 0.5 * tensor_Voigt(5), 0.5 * tensor_Voigt(4),
			0.5 * tensor_Voigt(5), tensor_Voigt(1), 0.5 * tensor_Voigt(3),
			0.5 * tensor_Voigt(4), 0.5 * tensor_Voigt(3), tensor_Voigt(2);
		return tensor;
	}


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


int main() {

	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_real_distribution<double> epsv_rand(0.0, 1.0);
	const double dt = 0.1;

	std::uniform_real_distribution<double> v_rand(-10.0, 10.0);


	Eigen::MatrixX<double> x0local =
		Eigen::MatrixX<double>::NullaryExpr(3, 4, [&]() { return v_rand(gen); });

	const double lambda = 6.0e5;
	const double mu = 0.0;

	Eigen::Matrix<double, 6, 6> c_Voigt;
	c_Voigt <<
		lambda + 2.0 * mu, lambda, 0, 0, 0, 0,
		lambda, lambda + 2.0 * mu, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0,
		0, 0, 0, mu, 0, 0,
		0, 0, 0, 0, mu, 0,
		0, 0, 0, 0, 0, mu;

	Eigen::VectorXd thicklocal(4);
	Eigen::MatrixXd V1local(3, 4);
	Eigen::MatrixXd V2local(3, 4);
	Eigen::MatrixXd Vnlocal(3, 4);
	Eigen::MatrixXd Vn0local(3, 4);

	thicklocal.setConstant(0.1);
	V1local <<
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 0.0;
	V2local <<
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 0.0;
	Vn0local <<
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0;
	Vnlocal <<
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 1.0;



	using AV_T = autodiff::var;
	int ivs_size = 4;

	GaussQuadrature::Quadrature gauss{};
	gauss = GaussQuadrature::hexa8;
	ivs_size = 4;

	autodiff::VectorXvar vs_in(5 * ivs_size);
	vs_in.setZero();
	for (int i = 0; i < vs_in.size(); ++i) {
		vs_in(i) = v_rand(gen);
	}


	//auto calc_phi = [&](double r, double s, double t, double W)->autodiff::var {

	//	Eigen::MatrixX<AV_T> xlocal(3, 4);

	//	xlocal.col(0) = vs_in.segment(0, 3);
	//	xlocal.col(1) = vs_in.segment(5, 3);
	//	xlocal.col(2) = vs_in.segment(10, 3);
	//	xlocal.col(3) = vs_in.segment(15, 3);

	//	Eigen::MatrixX<AV_T> Vnlocal_nextstep(3, 4);

	//	for (int i = 0; i < 4; ++i) {
	//		int idx_alph = i * 5 + 3;
	//		int idx_beta = i * 5 + 4;
	//		Vnlocal_nextstep.col(i) =
	//			Vnlocal.col(i) -
	//			vs_in(idx_alph) * V2local.col(i) + vs_in(idx_beta) * V1local.col(i) -
	//			0.5 * (vs_in(idx_alph) * vs_in(idx_alph) + vs_in(idx_beta) * vs_in(idx_beta)) * Vnlocal.col(i);
	//	}


	//	// MITC4
	//	AV_T tilde_e_rt_A, tilde_e_rt_C;
	//	AV_T tilde_e_st_B, tilde_e_st_D;
	//	//Eigen::Vector<AV_T, 20> B_NSX_rt_A, B_NSX_st_B, B_NSX_rt_C, B_NSX_st_D;

	//	// interpolation method
	//	{

	//		// tying point A
	//		{
	//			double r = 0.0; double s = 1.0; double t = 0.0;
	//			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
	//			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);
	//			//compute_dSdR(thicklocal, 0, N_xi, DN_xi, dSdr_A, dSds_A, dSdt_A);

	//			Eigen::MatrixX<AV_T> dXdR = MITC4::compute_derivative_position_3x3(t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
	//			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
	//			//Eigen::MatrixX<AV_T> dNdR = MITC4::compute_derivative_shape_displacement_20x9(
	//			//	t, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

	//			tilde_e_rt_A = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
	//			//tilde_e_rt_A = dhalfXpX0dR.col(0).dot(dXmX0dR.col(2)) + dhalfXpX0dR.col(2).dot(dXmX0dR.col(0));

	//			//B_NSX_rt_A =
	//			//	dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
	//			//	dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);

	//		}
	//		// tying point B
	//		{
	//			double r = -1.0; double s = 0.0; double t = 0.0;
	//			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
	//			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

	//			Eigen::MatrixX<AV_T> dXdR = MITC4::compute_derivative_position_3x3(t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
	//			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
	//			//Eigen::MatrixX<AV_T> dNdR = MITC4::compute_derivative_shape_displacement_20x9(
	//			//	t, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

	//			tilde_e_st_B = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
	//			//tilde_e_st_B = dhalfXpX0dR.col(1).dot(dXmX0dR.col(2)) + dhalfXpX0dR.col(2).dot(dXmX0dR.col(1));

	//			//B_NSX_st_B =
	//			//	dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
	//			//	dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
	//		}
	//		// tying point C
	//		{
	//			double r = 0.0; double s = -1.0; double t = 0.0;
	//			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
	//			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

	//			Eigen::MatrixX<AV_T> dXdR = MITC4::compute_derivative_position_3x3(t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
	//			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
	//			//Eigen::MatrixX<AV_T> dNdR = MITC4::compute_derivative_shape_displacement_20x9(
	//			//	t, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

	//			tilde_e_rt_C = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
	//			//tilde_e_rt_C = dhalfXpX0dR.col(0).dot(dXmX0dR.col(2)) + dhalfXpX0dR.col(2).dot(dXmX0dR.col(0));

	//			//B_NSX_rt_C =
	//			//	dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
	//			//	dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);
	//		}
	//		// tying point D
	//		{
	//			double r = 1.0; double s = 0.0; double t = 0.0;
	//			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
	//			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

	//			Eigen::MatrixX<AV_T> dXdR = MITC4::compute_derivative_position_3x3(t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
	//			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
	//			//Eigen::MatrixX<AV_T> dNdR = MITC4::compute_derivative_shape_displacement_20x9(
	//			//	t, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

	//			tilde_e_st_D = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
	//			//tilde_e_st_D = dhalfXpX0dR.col(1).dot(dXmX0dR.col(2)) + dhalfXpX0dR.col(2).dot(dXmX0dR.col(1));

	//			//B_NSX_st_D =
	//			//	dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
	//			//	dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
	//		}




	//	}



	//	//for (int igauss = 0; igauss < gauss.size; ++igauss)
	//	{

	//		//const auto& r = gauss.xi[3 * igauss + 0];
	//		//const auto& s = gauss.xi[3 * igauss + 1];
	//		//const auto& t = gauss.xi[3 * igauss + 2];
	//		//const auto& W = gauss.w[igauss];

	//		Eigen::VectorXd N_xi;
	//		Eigen::MatrixXd DN_xi, dNdR, dX0dR, dXmX0dR, dhalfXpX0dR;
	//		Eigen::MatrixX<AV_T> dXdR;
	//		{
	//			N_xi = MITC4::compute_shape_function_4x1(r, s);
	//			DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);
	//			dXdR = MITC4::compute_derivative_position_3x3(t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
	//			dX0dR = MITC4::compute_derivative_position_3x3(t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
	//		}


	//		// Green-Lagrange 변형률 텐서
	//		Eigen::Vector<AV_T, 6> Eij;
	//		Eij[0] = 0.5 * (dXdR.col(0).dot(dXdR.col(0)) - dX0dR.col(0).dot(dX0dR.col(0)));
	//		Eij[1] = 0.5 * (dXdR.col(1).dot(dXdR.col(1)) - dX0dR.col(1).dot(dX0dR.col(1)));
	//		Eij[2] = 0.5 * (dXdR.col(2).dot(dXdR.col(2)) - dX0dR.col(2).dot(dX0dR.col(2)));

	//		//MITC
	//		Eij[3] = 0.5 * (1.0 + r) * tilde_e_st_D + 0.5 * (1.0 - r) * tilde_e_st_B;
	//		Eij[4] = 0.5 * (1.0 + s) * tilde_e_rt_A + 0.5 * (1.0 - s) * tilde_e_rt_C;

	//		Eij[5] = dXdR.col(0).dot(dXdR.col(1)) - dX0dR.col(0).dot(dX0dR.col(1));





	//		Eigen::MatrixXd cglobal = c_Voigt;

	//		//{
	//		//	auto contra_basis = Soft::compute_contravariant(dX0dR.col(0), dX0dR.col(1), dX0dR.col(2));
	//		//	auto local_basis = Soft::compute_local_basis(dX0dR.col(0), dX0dR.col(1), dX0dR.col(2));

	//		//	auto trans_contravariant_to_local = Soft::compute_transform_matrix(
	//		//		contra_basis.col(0), contra_basis.col(1), contra_basis.col(2),
	//		//		local_basis.col(0), local_basis.col(1), local_basis.col(2));

	//		//	cglobal = trans_contravariant_to_local.transpose() * c_Voigt * trans_contravariant_to_local;
	//		//}


	//		// 2nd Piola-Kirchhoff 텐서
	//		Eigen::VectorX<AV_T> Sij_6x1 = cglobal * Eij;

	//		double Jx0_xi = std::abs(dX0dR.determinant());
	//		double dt2_absJ0_W = dt * dt * Jx0_xi * W;


	//		return 0.5 * Sij_6x1.dot(Eij) * dt2_absJ0_W;


	//	}

	//	//return energy;

	//};



	double rin, sin, tin, Win;

	auto calc_phi2 = [&](const Eigen::VectorXd& vsd_in)->double {

		Eigen::MatrixX<double> xlocal(3, ivs_size);
		Eigen::MatrixX<double> Vnlocal_nextstep(3, ivs_size);

		for (int i = 0; i < ivs_size; ++i) {
			xlocal.col(i) = vsd_in.segment(i * 5, 3);

			int idx_alph = i * 5 + 3;
			int idx_beta = i * 5 + 4;
			Vnlocal_nextstep.col(i) =
				Vnlocal.col(i) -
				vsd_in(idx_alph) * V2local.col(i) + vsd_in(idx_beta) * V1local.col(i);// -
			//0.5 * (vsd_in(idx_alph) * vsd_in(idx_alph) + vsd_in(idx_beta) * vsd_in(idx_beta)) * Vnlocal.col(i);
		}


		// MITC3
		double tilde_ert_coeff, tilde_est_coeff, tilde_ehat;
		//Eigen::Vector<double, 15> B_NSX_rt_coeff, B_NSX_st_coeff, B_NSX_hat;

		// MITC4
		double tilde_e_rt_A, tilde_e_rt_C;
		double tilde_e_st_B, tilde_e_st_D;
		Eigen::MatrixXd B_NSX_rt_A, B_NSX_st_B, B_NSX_rt_C, B_NSX_st_D;
		//Eigen::Vector<double, 20> B_NSX_rt_A, B_NSX_st_B, B_NSX_rt_C, B_NSX_st_D;

		//// interpolation method
		//if constexpr (ELEM == ElementType::TRI) {
		//	std::array<double, 6> tilde_e_rt_X, tilde_e_st_X;
		//	for (int i = 0; i < 6; ++i) {
		//		const auto& p = tying_points_MITC3P.col(i);
		//		const auto N_xi = MITC3P::compute_shape_function_3x1(p[0], p[1]);
		//		const auto DN_xi = MITC3P::compute_derivative_shape_function_2x3(p[0], p[1]);

		//		Eigen::MatrixX<double> dXdR = MITC3P::compute_derivative_position_3x3(
		//			0.0, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
		//		Eigen::MatrixXd dX0dR = MITC3P::compute_derivative_position_3x3(
		//			0.0, x0local, thicklocal, Vn0local, DN_xi, N_xi);

		//		tilde_e_rt_X[i] = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
		//		tilde_e_st_X[i] = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));

		//	}
		//	double coeff_tmp1 = 1.0 / 3.0 * tilde_e_rt_X[0] + 1.0 / 3.0 * tilde_e_st_X[0];

		//	tilde_ert_coeff = 2.0 / 3.0 * tilde_e_rt_X[1] - 1.0 / 3.0 * tilde_e_st_X[1] + coeff_tmp1;
		//	tilde_est_coeff = 2.0 / 3.0 * tilde_e_st_X[2] - 1.0 / 3.0 * tilde_e_rt_X[2] + coeff_tmp1;

		//	tilde_ehat = tilde_e_rt_X[5] - tilde_e_rt_X[3] - tilde_e_st_X[5] + tilde_e_st_X[4];

		//}
		//else if constexpr (ELEM == ElementType::QUAD) {

			// tying point A
		{
			double r = 0.0; double s = 1.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_rt_A = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
			B_NSX_rt_A =
				dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);

		}
		// tying point B
		{
			double r = -1.0; double s = 0.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_st_B = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
			B_NSX_st_B =
				dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
		}
		// tying point C
		{
			double r = 0.0; double s = -1.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_rt_C = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
			B_NSX_rt_C =
				dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);
		}
		// tying point D
		{
			double r = 1.0; double s = 0.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_st_D = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
			B_NSX_st_D =
				dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
		}




		//}



		//for (int igauss = 0; igauss < gauss.size; ++igauss)
		{

			//const auto& r = gauss.xi[3 * igauss + 0];
			//const auto& s = gauss.xi[3 * igauss + 1];
			//const auto& t = gauss.xi[3 * igauss + 2];
			//const auto& W = gauss.w[igauss];

			Eigen::VectorXd N_xi;
			Eigen::MatrixXd DN_xi, dX0dR, dNdR;
			Eigen::MatrixX<double> dXdR;
			//if constexpr (ELEM == ElementType::TRI) {
			//	N_xi = MITC3P::compute_shape_function_3x1(rin, sin);
			//	DN_xi = MITC3P::compute_derivative_shape_function_2x3(rin, sin);
			//	dXdR = MITC3P::compute_derivative_position_3x3(tin, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			//	dX0dR = MITC3P::compute_derivative_position_3x3(tin, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			//}
			//else if constexpr (ELEM == ElementType::QUAD) {
			N_xi = MITC4::compute_shape_function_4x1(rin, sin);
			DN_xi = MITC4::compute_derivative_shape_function_2x4(rin, sin);
			dXdR = MITC4::compute_derivative_position_3x3(tin, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			dX0dR = MITC4::compute_derivative_position_3x3(tin, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			dNdR = MITC4::compute_derivative_shape_displacement_20x9(
				tin, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);
			//}


			// Green-Lagrange 변형률 텐서 (natual, contravariant 좌표계에서)
			Eigen::Vector<double, 6> Eij_Voigt;
			Eij_Voigt[0] = 0.5 * (dXdR.col(0).dot(dXdR.col(0)) - dX0dR.col(0).dot(dX0dR.col(0)));
			Eij_Voigt[1] = 0.5 * (dXdR.col(1).dot(dXdR.col(1)) - dX0dR.col(1).dot(dX0dR.col(1)));
			Eij_Voigt[2] = 0.0;// 0.5 * (dXdR.col(2).dot(dXdR.col(2)) - dX0dR.col(2).dot(dX0dR.col(2)));

			//Eij_Voigt[3] = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
			//Eij_Voigt[4] = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));

			////MITC 보간
			//if constexpr (ELEM == ElementType::TRI) {
			//	Eij_Voigt[3] = tilde_est_coeff + tilde_ehat * (1.0 / 3.0 - rin);
			//	Eij_Voigt[4] = tilde_ert_coeff + tilde_ehat * (sin - 1.0 / 3.0);
			//}
			//else if constexpr (ELEM == ElementType::QUAD) {
			Eij_Voigt[3] = 0.5 * (1.0 + rin) * tilde_e_st_D + 0.5 * (1.0 - rin) * tilde_e_st_B;
			Eij_Voigt[4] = 0.5 * (1.0 + sin) * tilde_e_rt_A + 0.5 * (1.0 - sin) * tilde_e_rt_C;
			//}

			Eij_Voigt[5] = dXdR.col(0).dot(dXdR.col(1)) - dX0dR.col(0).dot(dX0dR.col(1));

			//auto Eij = Soft::Voigt_to_tensor(Eij_Voigt);
			//Eigen::Matrix3d dRdX0 = dX0dR.fullPivLu().inverse();
			//Eigen::Matrix3d Eij_glo = dRdX0.transpose() * Eij* dRdX0;
			//auto Eij_glo_Voigt = Soft::tensor_to_Voigt(Eij_glo);

			Eigen::MatrixXd cglobal;
			{
				auto local_basis = Soft::compute_local_basis(dX0dR.col(0), dX0dR.col(1), dX0dR.col(2));
				auto contra_basis = Soft::compute_contravariant(dX0dR.col(0), dX0dR.col(1), dX0dR.col(2));

				//auto Q = Soft::compute_transform_matrix(
				//	local_basis.col(0), local_basis.col(1), local_basis.col(2),
				//	Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ());

				// 변환행렬 (local 좌표계 -> natural, contravariant 좌표계)
				auto Q = Soft::compute_transform_matrix(
					local_basis.col(0), local_basis.col(1), local_basis.col(2),
					contra_basis.col(0), contra_basis.col(1), contra_basis.col(2));

				cglobal = Q.transpose() * c_Voigt * Q;
			}

			//Eigen::VectorX<double> Sij_6x1 = cglobal * Eij_glo_Voigt;
			// 
			// 2nd Piola-Kirchhoff 텐서 (natual, contravariant 좌표계에서)
			Eigen::VectorX<double> Sij_6x1 = cglobal * Eij_Voigt;

			double Jx0_xi = std::abs(dX0dR.determinant());
			double dt2_absJ0_W = dt * dt * Jx0_xi * Win;



			//return 0.5 * Sij_6x1.dot(Eij_glo_Voigt) * dt2_absJ0_W;
			return 0.5 * Sij_6x1.dot(Eij_Voigt) * dt2_absJ0_W;

		}

		};







	auto calc_phi3 = [&](const Eigen::VectorXd& vsd_in)->double {

		Eigen::MatrixX<double> xlocal(3, ivs_size);
		Eigen::MatrixX<double> Vnlocal_nextstep(3, ivs_size);

		for (int i = 0; i < ivs_size; ++i) {
			xlocal.col(i) = vsd_in.segment(i * 5, 3);

			int idx_alph = i * 5 + 3;
			int idx_beta = i * 5 + 4;
			Vnlocal_nextstep.col(i) =
				Vnlocal.col(i) -
				vsd_in(idx_alph) * V2local.col(i) + vsd_in(idx_beta) * V1local.col(i);// -
			//0.5 * (vsd_in(idx_alph) * vsd_in(idx_alph) + vsd_in(idx_beta) * vsd_in(idx_beta)) * Vnlocal.col(i);
		}


		// MITC3
		double tilde_ert_coeff, tilde_est_coeff, tilde_ehat;
		//Eigen::Vector<double, 15> B_NSX_rt_coeff, B_NSX_st_coeff, B_NSX_hat;

		// MITC4
		double tilde_e_rt_A, tilde_e_rt_C;
		double tilde_e_st_B, tilde_e_st_D;
		Eigen::MatrixXd B_NSX_rt_A, B_NSX_st_B, B_NSX_rt_C, B_NSX_st_D;
		Eigen::MatrixXd B_NN_rt_A, B_NN_st_B, B_NN_rt_C, B_NN_st_D;
		//Eigen::Vector<double, 20> B_NSX_rt_A, B_NSX_st_B, B_NSX_rt_C, B_NSX_st_D;

		//// interpolation method
		//if constexpr (ELEM == ElementType::TRI) {
		//	std::array<double, 6> tilde_e_rt_X, tilde_e_st_X;
		//	for (int i = 0; i < 6; ++i) {
		//		const auto& p = tying_points_MITC3P.col(i);
		//		const auto N_xi = MITC3P::compute_shape_function_3x1(p[0], p[1]);
		//		const auto DN_xi = MITC3P::compute_derivative_shape_function_2x3(p[0], p[1]);

		//		Eigen::MatrixX<double> dXdR = MITC3P::compute_derivative_position_3x3(
		//			0.0, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
		//		Eigen::MatrixXd dX0dR = MITC3P::compute_derivative_position_3x3(
		//			0.0, x0local, thicklocal, Vn0local, DN_xi, N_xi);

		//		tilde_e_rt_X[i] = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
		//		tilde_e_st_X[i] = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));

		//	}
		//	double coeff_tmp1 = 1.0 / 3.0 * tilde_e_rt_X[0] + 1.0 / 3.0 * tilde_e_st_X[0];

		//	tilde_ert_coeff = 2.0 / 3.0 * tilde_e_rt_X[1] - 1.0 / 3.0 * tilde_e_st_X[1] + coeff_tmp1;
		//	tilde_est_coeff = 2.0 / 3.0 * tilde_e_st_X[2] - 1.0 / 3.0 * tilde_e_rt_X[2] + coeff_tmp1;

		//	tilde_ehat = tilde_e_rt_X[5] - tilde_e_rt_X[3] - tilde_e_st_X[5] + tilde_e_st_X[4];

		//}
		//else if constexpr (ELEM == ElementType::QUAD) {

			// tying point A
		{
			double r = 0.0; double s = 1.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_rt_A = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
			B_NSX_rt_A =
				dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);
			B_NN_rt_A =
				dNdR.col(0) * dNdR.col(6).transpose() +
				dNdR.col(1) * dNdR.col(7).transpose() +
				dNdR.col(2) * dNdR.col(8).transpose() +
				dNdR.col(6) * dNdR.col(0).transpose() +
				dNdR.col(7) * dNdR.col(1).transpose() +
				dNdR.col(8) * dNdR.col(2).transpose();

		}
		// tying point B
		{
			double r = -1.0; double s = 0.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_st_B = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
			B_NSX_st_B =
				dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
			B_NN_st_B =
				dNdR.col(3) * dNdR.col(6).transpose() +
				dNdR.col(4) * dNdR.col(7).transpose() +
				dNdR.col(5) * dNdR.col(8).transpose() +
				dNdR.col(6) * dNdR.col(3).transpose() +
				dNdR.col(7) * dNdR.col(4).transpose() +
				dNdR.col(8) * dNdR.col(5).transpose();
		}
		// tying point C
		{
			double r = 0.0; double s = -1.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_rt_C = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));
			B_NSX_rt_C =
				dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);
			B_NN_rt_C =
				dNdR.col(0) * dNdR.col(6).transpose() +
				dNdR.col(1) * dNdR.col(7).transpose() +
				dNdR.col(2) * dNdR.col(8).transpose() +
				dNdR.col(6) * dNdR.col(0).transpose() +
				dNdR.col(7) * dNdR.col(1).transpose() +
				dNdR.col(8) * dNdR.col(2).transpose();
		}
		// tying point D
		{
			double r = 1.0; double s = 0.0; double t = 0.0;
			const auto N_xi = MITC4::compute_shape_function_4x1(r, s);
			const auto DN_xi = MITC4::compute_derivative_shape_function_2x4(r, s);

			Eigen::MatrixX<double> dXdR = MITC4::compute_derivative_position_3x3(
				t, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			Eigen::MatrixXd dX0dR = MITC4::compute_derivative_position_3x3(
				t, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			auto dNdR = MITC4::compute_derivative_shape_displacement_20x9
			(0, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);

			tilde_e_st_D = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
			B_NSX_st_D =
				dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
				dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
			B_NN_st_D =
				dNdR.col(3) * dNdR.col(6).transpose() +
				dNdR.col(4) * dNdR.col(7).transpose() +
				dNdR.col(5) * dNdR.col(8).transpose() +
				dNdR.col(6) * dNdR.col(3).transpose() +
				dNdR.col(7) * dNdR.col(4).transpose() +
				dNdR.col(8) * dNdR.col(5).transpose();
		}




		//}



		//for (int igauss = 0; igauss < gauss.size; ++igauss)
		{

			//const auto& r = gauss.xi[3 * igauss + 0];
			//const auto& s = gauss.xi[3 * igauss + 1];
			//const auto& t = gauss.xi[3 * igauss + 2];
			//const auto& W = gauss.w[igauss];

			Eigen::VectorXd N_xi;
			Eigen::MatrixXd DN_xi, dX0dR, dNdR;
			Eigen::MatrixX<double> dXdR;
			//if constexpr (ELEM == ElementType::TRI) {
			//	N_xi = MITC3P::compute_shape_function_3x1(rin, sin);
			//	DN_xi = MITC3P::compute_derivative_shape_function_2x3(rin, sin);
			//	dXdR = MITC3P::compute_derivative_position_3x3(tin, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			//	dX0dR = MITC3P::compute_derivative_position_3x3(tin, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			//}
			//else if constexpr (ELEM == ElementType::QUAD) {
			N_xi = MITC4::compute_shape_function_4x1(rin, sin);
			DN_xi = MITC4::compute_derivative_shape_function_2x4(rin, sin);
			dXdR = MITC4::compute_derivative_position_3x3(tin, xlocal, thicklocal, Vnlocal_nextstep, DN_xi, N_xi);
			dX0dR = MITC4::compute_derivative_position_3x3(tin, x0local, thicklocal, Vn0local, DN_xi, N_xi);
			dNdR = MITC4::compute_derivative_shape_displacement_20x9(
				tin, xlocal, thicklocal, V1local, V2local, Vnlocal_nextstep, DN_xi, N_xi);
			//}


			// Green-Lagrange 변형률 텐서 (natual, contravariant 좌표계에서)
			Eigen::Vector<double, 6> Eij_Voigt;
			Eij_Voigt[0] = 0.5 * (dXdR.col(0).dot(dXdR.col(0)) - dX0dR.col(0).dot(dX0dR.col(0)));
			Eij_Voigt[1] = 0.5 * (dXdR.col(1).dot(dXdR.col(1)) - dX0dR.col(1).dot(dX0dR.col(1)));
			Eij_Voigt[2] = 0.0;// 0.5 * (dXdR.col(2).dot(dXdR.col(2)) - dX0dR.col(2).dot(dX0dR.col(2)));

			//Eij_Voigt[3] = dXdR.col(1).dot(dXdR.col(2)) - dX0dR.col(1).dot(dX0dR.col(2));
			//Eij_Voigt[4] = dXdR.col(0).dot(dXdR.col(2)) - dX0dR.col(0).dot(dX0dR.col(2));

			////MITC 보간
			//if constexpr (ELEM == ElementType::TRI) {
			//	Eij_Voigt[3] = tilde_est_coeff + tilde_ehat * (1.0 / 3.0 - rin);
			//	Eij_Voigt[4] = tilde_ert_coeff + tilde_ehat * (sin - 1.0 / 3.0);
			//}
			//else if constexpr (ELEM == ElementType::QUAD) {
			Eij_Voigt[3] = 0.5 * (1.0 + rin) * tilde_e_st_D + 0.5 * (1.0 - rin) * tilde_e_st_B;
			Eij_Voigt[4] = 0.5 * (1.0 + sin) * tilde_e_rt_A + 0.5 * (1.0 - sin) * tilde_e_rt_C;
			//}

			Eij_Voigt[5] = dXdR.col(0).dot(dXdR.col(1)) - dX0dR.col(0).dot(dX0dR.col(1));

			//auto Eij = Soft::Voigt_to_tensor(Eij_Voigt);
			//Eigen::Matrix3d dRdX0 = dX0dR.fullPivLu().inverse();
			//Eigen::Matrix3d Eij_glo = dRdX0.transpose() * Eij* dRdX0;
			//auto Eij_glo_Voigt = Soft::tensor_to_Voigt(Eij_glo);

			Eigen::MatrixXd cglobal;
			{
				auto local_basis = Soft::compute_local_basis(dX0dR.col(0), dX0dR.col(1), dX0dR.col(2));
				auto contra_basis = Soft::compute_contravariant(dX0dR.col(0), dX0dR.col(1), dX0dR.col(2));

				//auto Q = Soft::compute_transform_matrix(
				//	local_basis.col(0), local_basis.col(1), local_basis.col(2),
				//	Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ());

				// 변환행렬 (local 좌표계 -> natural, contravariant 좌표계)
				auto Q = Soft::compute_transform_matrix(
					local_basis.col(0), local_basis.col(1), local_basis.col(2),
					contra_basis.col(0), contra_basis.col(1), contra_basis.col(2));

				cglobal = Q.transpose() * c_Voigt * Q;
			}

			//Eigen::VectorX<double> Sij_6x1 = cglobal * Eij_glo_Voigt;
			// 
			// 2nd Piola-Kirchhoff 텐서 (natual, contravariant 좌표계에서)
			Eigen::VectorX<double> Sij_6x1 = cglobal * Eij_Voigt;

			double Jx0_xi = std::abs(dX0dR.determinant());
			double dt2_absJ0_W = dt * dt * Jx0_xi * Win;



			Eigen::MatrixXd A_NSx;
			{
				A_NSx.resize(20, 6);
				A_NSx.col(0) = dNdR.col(0) * dXdR(0, 0) + dNdR.col(1) * dXdR(1, 0) + dNdR.col(2) * dXdR(2, 0);
				A_NSx.col(1) = dNdR.col(3) * dXdR(0, 1) + dNdR.col(4) * dXdR(1, 1) + dNdR.col(5) * dXdR(2, 1);
				A_NSx.col(2).setZero();// = dNdR.col(6) * dXdR(0, 2) + dNdR.col(7) * dXdR(1, 2) + dNdR.col(8) * dXdR(2, 2);

				//A_NSx.col(3) =
				//	dNdR.col(3) * dXdR(0, 2) + dNdR.col(4) * dXdR(1, 2) + dNdR.col(5) * dXdR(2, 2) +
				//	dNdR.col(6) * dXdR(0, 1) + dNdR.col(7) * dXdR(1, 1) + dNdR.col(8) * dXdR(2, 1);
				//A_NSx.col(4) =
				//	dNdR.col(0) * dXdR(0, 2) + dNdR.col(1) * dXdR(1, 2) + dNdR.col(2) * dXdR(2, 2) +
				//	dNdR.col(6) * dXdR(0, 0) + dNdR.col(7) * dXdR(1, 0) + dNdR.col(8) * dXdR(2, 0);

				////MITC
				A_NSx.col(3) = (0.5 * (1.0 + rin)) * B_NSX_st_D + (0.5 * (1.0 - rin)) * B_NSX_st_B;
				A_NSx.col(4) = (0.5 * (1.0 + sin)) * B_NSX_rt_A + (0.5 * (1.0 - sin)) * B_NSX_rt_C;

				A_NSx.col(5) =
					dNdR.col(0) * dXdR(0, 1) + dNdR.col(1) * dXdR(1, 1) + dNdR.col(2) * dXdR(2, 1) +
					dNdR.col(3) * dXdR(0, 0) + dNdR.col(4) * dXdR(1, 0) + dNdR.col(5) * dXdR(2, 0);
			}

			Eigen::VectorXd grad = A_NSx * Sij_6x1 * dt2_absJ0_W;


			std::cout << "grad numeric" << std::endl;
			std::cout << grad << std::endl;


			Eigen::MatrixXd hess = A_NSx * cglobal * A_NSx.transpose() * dt2_absJ0_W;


			hess +=
				(
					(dNdR.col(0) * dNdR.col(0).transpose() +
						dNdR.col(1) * dNdR.col(1).transpose() +
						dNdR.col(2) * dNdR.col(2).transpose()) * Sij_6x1(0) +
					(dNdR.col(3) * dNdR.col(3).transpose() +
						dNdR.col(4) * dNdR.col(4).transpose() +
						dNdR.col(5) * dNdR.col(5).transpose()) * Sij_6x1(1) +
					//(dNdR.col(6) * dNdR.col(6).transpose() +
					//	dNdR.col(7) * dNdR.col(7).transpose() +
					//	dNdR.col(8) * dNdR.col(8).transpose()) * Sij_6x1(2) +
					//(dNdR.col(3) * dNdR.col(6).transpose() +
					//	dNdR.col(4) * dNdR.col(7).transpose() +
					//	dNdR.col(5) * dNdR.col(8).transpose() +
					//	dNdR.col(6) * dNdR.col(3).transpose() +
					//	dNdR.col(7) * dNdR.col(4).transpose() +
					//	dNdR.col(8) * dNdR.col(5).transpose()) * Sij_6x1(3) +
					//(dNdR.col(0) * dNdR.col(6).transpose() +
					//	dNdR.col(1) * dNdR.col(7).transpose() +
					//	dNdR.col(2) * dNdR.col(8).transpose() +
					//	dNdR.col(6) * dNdR.col(0).transpose() +
					//	dNdR.col(7) * dNdR.col(1).transpose() +
					//	dNdR.col(8) * dNdR.col(2).transpose()) * Sij_6x1(4) +
					((0.5 * (1.0 + rin)) * B_NN_st_D + (0.5 * (1.0 - rin)) * B_NN_st_B) * Sij_6x1(3) +
					((0.5 * (1.0 + sin)) * B_NN_rt_A + (0.5 * (1.0 - sin)) * B_NN_rt_C) * Sij_6x1(4) +
					(dNdR.col(0) * dNdR.col(3).transpose() +
						dNdR.col(1) * dNdR.col(4).transpose() +
						dNdR.col(2) * dNdR.col(5).transpose() +
						dNdR.col(3) * dNdR.col(0).transpose() +
						dNdR.col(4) * dNdR.col(1).transpose() +
						dNdR.col(5) * dNdR.col(2).transpose()) * Sij_6x1(5)
					) * dt2_absJ0_W;


			std::cout << "hess numeric" << std::endl;
			std::cout << hess << std::endl;

			//return 0.5 * Sij_6x1.dot(Eij_glo_Voigt) * dt2_absJ0_W;
			return 0.5 * Sij_6x1.dot(Eij_Voigt) * dt2_absJ0_W;

		}

		};








	Eigen::VectorXd fd_in(20);
	for (int i = 0; i < fd_in.size(); ++i) {
		fd_in(i) = val(vs_in(i));
	}

	//for (int igauss = 0; igauss < gauss.size; ++igauss)
	for (int igauss = 0; igauss < 1; ++igauss)
	{

		rin = gauss.xi[3 * igauss + 0];
		sin = gauss.xi[3 * igauss + 1];
		tin = gauss.xi[3 * igauss + 2];
		Win = gauss.w[igauss];

		//autodiff::var u = calc_phi(rin, sin, tin, Win);
		//Eigen::VectorXd grad;
		//auto hess = autodiff::hessian(u, vs_in, grad);

		//std::cout << "val : " << u << std::endl;
		//std::cout << grad << std::endl;
		//std::cout << hess << std::endl;

		Eigen::VectorXd grad2;
		Eigen::MatrixXd hess2;
		fd::finite_gradient(fd_in, calc_phi2, grad2, fd::AccuracyOrder::FOURTH);
		fd::finite_hessian(fd_in, calc_phi2, hess2, fd::AccuracyOrder::FOURTH);

		calc_phi3(fd_in);

		std::cout << "grad finite difference" << std::endl;
		std::cout << grad2 << std::endl;
		std::cout << "hess finite difference" << std::endl;
		std::cout << hess2 << std::endl;

	}





}