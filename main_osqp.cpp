#include <TinyAD/Scalar.hh>
#include <TinyAD/ScalarFunction.hh>
#include <cstdlib>
#include <cassert>
#include <tuple>
#include <array>
#include <set>
#include <iostream>
#include <format>
#include <ostream>
#include <sstream>
#include <fstream>
#include <string>
#include <random>


#include "osqp.h"


using vec3 = Eigen::Vector3d;
using veci3 = Eigen::Vector3i;
using veci4 = Eigen::Vector4i;

struct Mesh {

	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> pos, pos0;
	Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> c2v, c2f;
	Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> f2v, f2e;
	Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> e2v;

	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> vel;

};

using Point = Eigen::Vector3d;
using vec3 = Eigen::Vector3d;
struct Triangle {
	vec3 a, b, c;
};
struct Plane {
	vec3 n;
	double d;
};
struct Segment {
	vec3 a, b;
};
struct Ray {
	vec3 p, n;
};



vec3 closest(const vec3& p, const Plane& plane) {
	double t = (p.dot(plane.n) - plane.d) / plane.n.squaredNorm();
	return p - t * plane.n;
}
bool is_inside(const Point& p, const Plane& plane) {
	double dot = p.dot(plane.n);
	return std::abs(dot - plane.d) <= 1.e-12;
}
std::tuple<bool, double, double> raycast(const Ray& ray0, const Ray& ray1) {
	Eigen::Matrix3d mat;
	mat.col(0) = ray1.p - ray0.p;
	mat.col(1) = ray1.n;
	mat.col(2) = ray0.n.cross(ray1.n);
	double d1d2 = mat.col(2).squaredNorm();
	if (d1d2 < 1.e-12) {
		// 광선 겹침
		if (
			std::abs(mat.col(0).normalized().dot(ray0.n) - 1.0) < 1.e-12 &&
			std::abs(mat.col(0).normalized().dot(ray1.n) - 1.0) < 1.e-12
			) {
			return std::make_tuple(true, 0.0, 0.0);
		}
		return std::make_tuple(false, -1.0, -1.0);
	}
	double s = mat.determinant() / d1d2;
	if (s < 0.0) return std::make_tuple(false, -1.0, -1.0);
	mat.col(1) = ray0.n;
	double t = mat.determinant() / d1d2;
	if (t < 0.0) return std::make_tuple(false, -1.0, -1.0);
	return std::make_tuple(true, s, t);
}
bool is_inside(
	const Point& p,
	const Triangle& tri
) {
	// 우선 점이 삼각형 평면 위에 있는지 확인
	Plane plane;
	plane.n = (tri.b - tri.a).cross(tri.c - tri.a).normalized();
	plane.d = plane.n.dot(tri.a);
	if (is_inside(p, plane) == false) return false;

	vec3 pa = tri.a - p;
	vec3 pb = tri.b - p;
	vec3 pc = tri.c - p;

	vec3 u = pb.cross(pc);
	vec3 v = pc.cross(pa);
	if (u.dot(v) < 0.0) return false;
	vec3 w = pa.cross(pb);
	if (u.dot(w) < 0.0) return false;
	return true;
}
vec3 closest(const vec3& p, const Segment& seg) {
	vec3 ab = seg.b - seg.a;
	double t = (p - seg.a).dot(ab) / ab.squaredNorm();
	t = std::clamp(t, 0.0, 1.0);
	return seg.a + t * ab;
}
vec3 closest(
	const Point& p,
	const Triangle& tri
) {
	Plane plane;
	plane.n = (tri.b - tri.a).cross(tri.c - tri.a).normalized();
	plane.d = plane.n.dot(tri.a);
	Point cloPt = closest(p, plane);
	double magSq0 = (p - cloPt).squaredNorm();
	if (is_inside(cloPt, tri)) return cloPt;

	Point c1 = closest(p, Segment(tri.a, tri.b));
	Point c2 = closest(p, Segment(tri.b, tri.c));
	Point c3 = closest(p, Segment(tri.c, tri.a));

	double magSq1 = (p - c1).squaredNorm();
	double magSq2 = (p - c2).squaredNorm();
	double magSq3 = (p - c3).squaredNorm();

	if (magSq1 < magSq2 && magSq1 < magSq3) {
		return c1;
	}
	else if (magSq2 < magSq1 && magSq2 < magSq3) {
		return c2;
	}
	return c3;
}

std::tuple<double, vec3, vec3> closest_distance(const Point& p, const Triangle& tri) {
	auto p_clo = closest(p, tri);
	double dist = (p - p_clo).squaredNorm();
	return std::make_tuple(dist, p, p_clo);
}
std::tuple<double, vec3, vec3> closest_distance(const Segment& seg0, const Segment& seg1) {
	double dist = std::numeric_limits<double>::infinity();
	vec3 v0, v1;
	vec3 v0_tar = seg0.a;
	vec3 v1_tar = closest(v0_tar, seg1);
	double dist_tar = (v0_tar - v1_tar).squaredNorm();
	if (dist_tar < dist) {
		dist = dist_tar; v0 = v0_tar; v1 = v1_tar;
	}
	v0_tar = seg0.b;
	v1_tar = closest(v0_tar, seg1);
	dist_tar = (v0_tar - v1_tar).squaredNorm();
	if (dist_tar < dist) {
		dist = dist_tar; v0 = v0_tar; v1 = v1_tar;
	}
	v1_tar = seg1.a;
	v0_tar = closest(v1_tar, seg0);
	dist_tar = (v0_tar - v1_tar).squaredNorm();
	if (dist_tar < dist) {
		dist = dist_tar; v0 = v0_tar; v1 = v1_tar;
	}
	v1_tar = seg1.b;
	v0_tar = closest(v1_tar, seg0);
	dist_tar = (v0_tar - v1_tar).squaredNorm();
	if (dist_tar < dist) {
		dist = dist_tar; v0 = v0_tar; v1 = v1_tar;
	}

	//Ray ray0, ray1;
	//ray0.p = seg0.a;
	//ray0.n = (seg0.b - seg0.a).normalized();
	//ray1.p = seg1.a;
	//ray1.n = (seg1.b - seg1.a).normalized();
	//auto [is_collide, s, t] = raycast(ray0, ray1);

	//if (!is_collide && 
	//	s * s <= (seg0.b - seg0.a).squaredNorm() &&
	//	t * t <= (seg1.b - seg1.a).squaredNorm()) {
	//	dist = std::min(dist, ((seg0.b - seg0.a) * s - (seg1.b - seg1.a) * t).squaredNorm());
	//}
	return std::make_tuple(dist, v0, v1);
}



double barrier(const double d, const double dhat)
{
	if (d <= 0.0) {
		return std::numeric_limits<double>::infinity();
	}
	if (d >= dhat) {
		return 0;
	}
	// b(d) = -(d-d̂)²ln(d / d̂)
	const double d_minus_dhat = (d - dhat);
	return -d_minus_dhat * d_minus_dhat * log(d / dhat);
}

double barrier_first_derivative(const double d, const double dhat)
{
	if (d <= 0.0 || d >= dhat) {
		return 0.0;
	}
	// b(d) = -(d - d̂)²ln(d / d̂)
	// b'(d) = -2(d - d̂)ln(d / d̂) - (d-d̂)²(1 / d)
	//       = (d - d̂) * (-2ln(d/d̂) - (d - d̂) / d)
	//       = (d̂ - d) * (2ln(d/d̂) - d̂/d + 1)
	return (dhat - d) * (2 * log(d / dhat) - dhat / d + 1);
}

double barrier_second_derivative(const double d, const double dhat)
{
	if (d <= 0.0 || d >= dhat) {
		return 0.0;
	}
	const double dhat_d = dhat / d;
	return (dhat_d + 2) * dhat_d - 2 * log(d / dhat) - 3;
}




void make_vtu(Mesh& mesh, std::string filename) {


	std::ofstream outputFile;

	std::string saveFormat = "ascii";

	outputFile.open(filename);
	if (outputFile.fail()) {
		std::cerr << "Unable to write file for writing." << std::endl;
	}
	outputFile << "<?xml version=\"1.0\"?>" << std::endl;

	outputFile << " <VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << std::endl;
	outputFile << "  <UnstructuredGrid>" << std::endl;

	// vtk 파일 만들기
	// Field data
	outputFile << "    <FieldData>" << std::endl;
	outputFile << "    </FieldData>" << std::endl;

	outputFile << "   <Piece NumberOfPoints=\"" <<
		mesh.pos.rows() <<
		"\" NumberOfCells=\"" <<
		mesh.c2v.rows() <<
		// mesh.cells.size() + 1 << 
		"\">" << std::endl;

	// Points data
	outputFile << "    <PointData>" << std::endl;
	outputFile << "    </PointData>" << std::endl;



	// Cells data
	outputFile << "    <CellData>" << std::endl;
	outputFile << "    </CellData>" << std::endl;

	// Points
	outputFile << "    <Points>" << std::endl;
	{
		outputFile << "     <DataArray type=\"Float64\" Name=\"NodeCoordinates\" NumberOfComponents=\"3\" format=\"" << saveFormat << "\">" << std::endl;
		for (auto p : mesh.pos.rowwise()) {
			outputFile << p[0] << " " << p[1] << " " << p[2] << std::endl;
		}
		outputFile << "     </DataArray>" << std::endl;
	}
	outputFile << "   </Points>" << std::endl;





	// cells
	outputFile << "   <Cells>" << std::endl;
	// connectivity (cell's points)
	{
		outputFile << "    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"" << saveFormat << "\">" << std::endl;
		for (auto ivs : mesh.c2v.rowwise()) {
			for (auto& iv : ivs) {
				outputFile << iv << " ";
			}
			outputFile << std::endl;
		}
		outputFile << "     </DataArray>" << std::endl;
	}


	// offsets (cell's points offset)
	{
		outputFile << "    <DataArray type=\"Int32\" Name=\"offsets\" format=\"" << saveFormat << "\">" << std::endl;
		int cellFaceOffset = 0;
		for (auto ivs : mesh.c2v.rowwise()) {
			cellFaceOffset += ivs.cols();
			outputFile << cellFaceOffset << " ";
		}
		outputFile << std::endl;
		outputFile << "     </DataArray>" << std::endl;
	}

	// types (cell's type, 42 = polyhedron)
	{
		outputFile << "    <DataArray type=\"UInt8\" Name=\"types\" format=\"" << saveFormat << "\">" << std::endl;
		for (auto ivs : mesh.c2v.rowwise()) {
			outputFile << "42" << " ";
		}
		outputFile << std::endl;
		outputFile << "     </DataArray>" << std::endl;
	}


	// faces (cell's faces number, each face's point number, cell's faces's points)
	{
		outputFile << "    <DataArray type=\"Int32\" IdType=\"1\" Name=\"faces\" format=\"" << saveFormat << "\">" << std::endl;
		for (auto ifs : mesh.c2f.rowwise()) {
			outputFile << ifs.cols() << " ";
			for (auto& f : ifs) {
				auto ivs = mesh.f2v.row(f);
				outputFile << ivs.cols() << " ";
				for (auto& j : ivs) {
					outputFile << j << " ";
				}
			}
		}
		outputFile << std::endl;
		outputFile << "     </DataArray>" << std::endl;
	}


	// faceoffsets (cell's face offset)
	{
		outputFile << "    <DataArray type=\"Int32\" IdType=\"1\" Name=\"faceoffsets\" format=\"" << saveFormat << "\">" << std::endl;
		int cellFacePointOffset = 0;
		for (auto ifs : mesh.c2f.rowwise()) {
			int numbering = 1 + ifs.cols();
			for (auto& f : ifs) {
				auto ivs = mesh.f2v.row(f);
				numbering += ivs.cols();
			}
			cellFacePointOffset += numbering;
			outputFile << cellFacePointOffset << " ";
		}
		outputFile << std::endl;
		outputFile << "     </DataArray>" << std::endl;
	}

	outputFile << std::endl;
	outputFile << "   </Cells>" << std::endl;


	outputFile << "  </Piece>" << std::endl;
	outputFile << " </UnstructuredGrid>" << std::endl;


	outputFile << "</VTKFile>" << std::endl;


	outputFile.close();


}



int main() {


	Mesh mesh0, mesh1;

	auto make_tet4 = [](Mesh& mesh) {
		mesh.pos.resize(4, Eigen::NoChange);
		mesh.vel.resize(4, Eigen::NoChange);
		mesh.c2v.resize(1, Eigen::NoChange);
		mesh.c2f.resize(1, Eigen::NoChange);
		mesh.f2v.resize(4, Eigen::NoChange);
		mesh.pos.row(0) = vec3(0, 0, 0);
		mesh.pos.row(1) = vec3(0, 1, 0);
		mesh.pos.row(2) = vec3(0, 0, 1);
		mesh.pos.row(3) = vec3(1, 0, 0);
		mesh.c2v.row(0) = veci4(0, 1, 2, 3);
		mesh.c2f.row(0) = veci4(0, 1, 2, 3);
		mesh.f2v.row(0) = veci3(0, 1, 2);
		mesh.f2v.row(1) = veci3(0, 1, 3);
		mesh.f2v.row(2) = veci3(0, 2, 3);
		mesh.f2v.row(3) = veci3(1, 2, 3);
		mesh.pos0 = mesh.pos;
		mesh.vel.setZero();
	};


	auto setup_f2v_to_e2v_f2e = [](Mesh& mesh) {

		auto comp = [](auto& a, auto& b) {
			if (a[0] != b[0]) return a[0] < b[0];
			return a[1] < b[1];
			};

		std::set<std::array<int, 3>, decltype(comp)> e_tmp(comp);
		mesh.f2e.resize(mesh.f2v.rows(), Eigen::NoChange);
		auto max_index = std::numeric_limits<int>::max();
		mesh.f2e.setConstant(max_index);
		for (int i = 0; auto ivs : mesh.f2v.rowwise()) {
			for (int j = 0; j < ivs.size(); ++j) {
				auto& fir = ivs[j];
				auto& sec = ivs[(j + 1) % ivs.size()];

				// vertex idx 오름차순으로 정리
				std::array<int, 3> input_value;
				if (fir < sec) {
					input_value = { fir, sec, max_index };
				}
				else {
					input_value = { sec, fir, max_index };
				}
				auto res = e_tmp.find(input_value);

				if (res != e_tmp.end()) {
					// 값이 있을 때
					if (mesh.f2e.row(i)[0] == max_index) {
						mesh.f2e.row(i)[0] = (*res)[2];
					}
					else {
						mesh.f2e.row(i)[1] = (*res)[2];
					}
				}
				else {
					// 값이 없을 때
					if (mesh.f2e.row(i)[0] == max_index) {
						mesh.f2e.row(i)[0] = e_tmp.size();
					}
					else {
						mesh.f2e.row(i)[1] = e_tmp.size();
					}
					input_value[2] = e_tmp.size();
					e_tmp.insert(input_value);
				}

			}
			++i;
		}
		mesh.e2v.resize(e_tmp.size(), Eigen::NoChange);
		for (auto& item : e_tmp) {
			mesh.e2v.row(item[2])[0] = item[0];
			mesh.e2v.row(item[2])[1] = item[1];
		}

	};

	make_tet4(mesh0);
	make_tet4(mesh1);
	for (auto i = 0; i < mesh1.pos.rows(); ++i) {
		mesh1.pos.row(i) += vec3(0.0, 1.5, 0.0);
		mesh1.pos0.row(i) += vec3(0.0, 1.5, 0.0);
	}

	setup_f2v_to_e2v_f2e(mesh0);
	setup_f2v_to_e2v_f2e(mesh1);


	make_vtu(mesh0, "mesh0_0.vtu");
	make_vtu(mesh1, "mesh1_0.vtu");


	double dt = 0.1;
	double density = 100.0;

	auto jacobian_tet4 = [](
		const Eigen::Vector3d& x0,
		const Eigen::Vector3d& x1,
		const Eigen::Vector3d& x2,
		const Eigen::Vector3d& x3
		) {
			Eigen::Matrix3d J;
			J.col(0) = x1 - x0;
			J.col(1) = x2 - x0;
			J.col(2) = x3 - x0;
			return J;
		};

	//const int quad_ngauss = 5;
	//const Eigen::Matrix<double, 5, 3> quad_chi =
	//	(Eigen::Matrix<double, 5, 3>() <<
	//		0.25, 0.25, 0.25,
	//		1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0,
	//		0.5, 1.0 / 6.0, 1.0 / 6.0,
	//		1.0 / 6.0, 0.5, 1.0 / 6.0,
	//		1.0 / 6.0, 1.0 / 6.0, 0.5
	//		).finished();
	//const Eigen::Vector<double, 5> quad_w(
	//	-0.1333333333333333, 0.075, 0.075, 0.075, 0.075
	//);


	const int quad_ngauss = 4;
	const Eigen::Matrix<double, quad_ngauss, 3> quad_chi =
		(Eigen::Matrix<double, quad_ngauss, 3>() <<
			0.58541020, 0.13819660, 0.13819660,
			0.13819660, 0.58541020, 0.13819660,
			0.13819660, 0.13819660, 0.58541020,
			0.13819660, 0.13819660, 0.13819660
			).finished();
	const Eigen::Vector<double, quad_ngauss> quad_w(
		0.041666667, 0.041666667, 0.041666667, 0.041666667
	);


	auto N_q = [](double xi, double eta, double zeta) {
		return
			Eigen::Vector4d(1.0 - eta - zeta - xi, xi, eta, zeta);
		};
	auto dNdxi_q = [](double xi, double eta, double zeta) {
		Eigen::Matrix<double, 3, 4> mat;
		mat << -1.0, 1.0, 0.0, 0.0,
			-1.0, 0.0, 1.0, 0.0,
			-1.0, 0.0, 0.0, 1.0;
		return mat;
		};


	Eigen::MatrixXd hessian0, hessian1;
	Eigen::VectorXd gradient0, gradient1;
	hessian0.resize(4 * 3, 4 * 3); hessian0.setZero();
	hessian1.resize(4 * 3, 4 * 3); hessian1.setZero();
	gradient0.resize(4 * 3); gradient0.setZero();
	gradient1.resize(4 * 3); gradient1.setZero();



	double mu_ = 26.0;
	double lambda_ = 58.0;



	//auto ela_poten_const = [mu_, lambda_](const vec3& p0, const vec3& v1) {
	//	using ADouble = TinyAD::Scalar<3, double, true>;
	//	Eigen::Vector<ADouble, 3> x_AD = ADouble::make_active(p0);
	//	ADouble dist = (x_AD - v1).squaredNorm();
	//	double Jx_xi = std::abs(Dx_xi.determinant());
	//	Eigen::MatrixXd F = xlocal * DN_x0.transpose();
	//	double J = F.determinant();
	//	Eigen::MatrixXd b = F * F.transpose();
	//	double Ib = b.trace(); // Frobenius norm squared

	//	ADouble potential =
	//		0.5 * mu_ * (Ib - 3.0) - mu_ * logJ +
	//		0.5 * lambda_ * logJ * logJ;
	//	return potential;
	//};


	auto inertia_elasticity_const = [
			&quad_chi = quad_chi,
			&N_q = N_q,
			&dNdxi_q = dNdxi_q,
			&density = density,
			&quad_w = quad_w,
			&mu_ = mu_,
			&dt = dt,
			&lambda_ = lambda_
	](Mesh& mesh, Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian) {

		for (int i = 0; i < mesh.c2v.rows(); ++i) {
			auto ivs = mesh.c2v.row(i);

			Eigen::Matrix<double, 3, 4> x0local, xlocal, vlocal;
			x0local.col(0) = mesh.pos0.row(ivs[0]);
			x0local.col(1) = mesh.pos0.row(ivs[1]);
			x0local.col(2) = mesh.pos0.row(ivs[2]);
			x0local.col(3) = mesh.pos0.row(ivs[3]);

			xlocal.col(0) = mesh.pos.row(ivs[0]);
			xlocal.col(1) = mesh.pos.row(ivs[1]);
			xlocal.col(2) = mesh.pos.row(ivs[2]);
			xlocal.col(3) = mesh.pos.row(ivs[3]);

			vlocal.col(0) = mesh.vel.row(ivs[0]);
			vlocal.col(1) = mesh.vel.row(ivs[1]);
			vlocal.col(2) = mesh.vel.row(ivs[2]);
			vlocal.col(3) = mesh.vel.row(ivs[3]);


			Eigen::Matrix<double, 3, 3> X0;
			X0.col(0) = mesh.pos0.row(ivs[1]) - mesh.pos0.row(ivs[0]);
			X0.col(1) = mesh.pos0.row(ivs[2]) - mesh.pos0.row(ivs[0]);
			X0.col(2) = mesh.pos0.row(ivs[3]) - mesh.pos0.row(ivs[0]);
			Eigen::Matrix<double, 3, 3> restTriInv = X0.inverse();


			//Eigen::Matrix<double, 3, 4> gradient_tot = Eigen::Matrix<double, 3, 4>::Zero();
			//Eigen::Matrix<double, 3, 4> hessian_tot = Eigen::Matrix<double, 3, 4>::Zero();

			for (int igauss = 0; igauss < quad_ngauss; ++igauss) {
				auto xi = quad_chi.row(igauss);

				auto N_i = N_q(xi[0], xi[1], xi[2]);
				auto DN_xi = dNdxi_q(xi[0], xi[1], xi[2]);

				Eigen::MatrixXd Dx0_xi = x0local * DN_xi.transpose(); // 3 x 3
				Eigen::MatrixXd Dx_xi = xlocal * DN_xi.transpose(); // 3 x 3

				Eigen::MatrixXd DN_x0 = Dx0_xi.transpose().inverse() * DN_xi; // 3 x 4
				Eigen::MatrixXd DN_x = Dx_xi.transpose().inverse() * DN_xi; // 3 x 4

				//std::cout << DN_x << std::endl;

				double Jx_xi = std::abs(Dx_xi.determinant());
				Eigen::MatrixXd F = xlocal * DN_x0.transpose();
				double J = F.determinant();
				Eigen::MatrixXd b = F * F.transpose();
				double Ib = b.trace(); // Frobenius norm squared
				double logJ = std::log(J);


				//Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(b);
				//lambda = eigensolver.eigenvalues().cwiseSqrt();
				//n = eigensolver.eigenvectors();

				double JW = quad_w[igauss] * Jx_xi;

				Eigen::Matrix<double, 3, 1> sigma;
				Eigen::Matrix<double, 3, 3> U;
				Eigen::Matrix<double, 3, 3> V;
				{
					Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(F);
					sigma = eigenSolver.eigenvalues();
					for (int i = 0; i < sigma.size(); ++i) {
						sigma(i) = std::max(sigma(i), 1.e-12);
					}
					U = eigenSolver.eigenvectors();
					V = eigenSolver.eigenvectors().transpose();
				}
				//eigenSolver.eigenvectors() * eigenvalues.asDiagonal() * eigenSolver.eigenvectors().transpose();

				Eigen::Matrix<double, 3, 1> dE_div_dsigma;
				Eigen::Matrix<double, 3, 3> d2E_div_dsigma2;

				//if (u == 0.0 && lambda == 0.0) {
				//	dE_div_dsigma.setZero();
				//	d2E_div_dsigma2.setZero();
				//}

				const double log_sigmaProd = std::log(sigma.prod());

				// 제1 Piola-Kirchhoff 응력 텐서 P = ∂Ψ / ∂F = μ F + (λ log(J) - μ) F ^ (-T)
				for (int idim = 0; idim < 3; ++idim) {
					const double inv0 = 1.0 / sigma[idim];
					dE_div_dsigma[idim] = mu_ * (sigma[idim] - inv0) + lambda_ * inv0 * log_sigmaProd;
				}

				// compute A
				const double inv2_0 = 1.0 / sigma[0] / sigma[0];
				d2E_div_dsigma2(0, 0) = mu_ * (1.0 + inv2_0) - lambda_ * inv2_0 * (log_sigmaProd - 1.0);
				const double inv2_1 = 1.0 / sigma[1] / sigma[1];
				d2E_div_dsigma2(1, 1) = mu_ * (1.0 + inv2_1) - lambda_ * inv2_1 * (log_sigmaProd - 1.0);
				d2E_div_dsigma2(0, 1) = d2E_div_dsigma2(1, 0) = lambda_ / sigma[0] / sigma[1];
				const double inv2_2 = 1.0 / sigma[2] / sigma[2];
				d2E_div_dsigma2(2, 2) = mu_ * (1.0 + inv2_2) - lambda_ * inv2_2 * (log_sigmaProd - 1.0);
				d2E_div_dsigma2(1, 2) = d2E_div_dsigma2(2, 1) = lambda_ / sigma[1] / sigma[2];
				d2E_div_dsigma2(2, 0) = d2E_div_dsigma2(0, 2) = lambda_ / sigma[2] / sigma[0];


				Eigen::VectorXd eigenval;
				{
					Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(d2E_div_dsigma2);
					eigenval = eigenSolver.eigenvalues();
					for (int i = 0; i < eigenval.size(); ++i) {
						eigenval(i) = std::max(eigenval(i), 1.e-12);
					}
					d2E_div_dsigma2 = 
						eigenSolver.eigenvectors() * 
						eigenval.asDiagonal() *
						eigenSolver.eigenvectors().transpose();

				}


				// compute B
				const int Cdim2 = 3;// dim* (dim - 1) / 2;
				Eigen::Matrix<double, Cdim2, 1> BLeftCoef;
				//TODO: right coef also has analytical form
				const double sigmaProd = eigenval.prod();
				const double middle = mu_ - lambda_ * std::log(sigmaProd);
				BLeftCoef[0] = (mu_ + middle / eigenval[0] / eigenval[1]) / 2.0;
				BLeftCoef[1] = (mu_ + middle / eigenval[1] / eigenval[2]) / 2.0;
				BLeftCoef[2] = (mu_ + middle / eigenval[2] / eigenval[0]) / 2.0;

				Eigen::Matrix2d B[Cdim2];
				for (int cI = 0; cI < Cdim2; cI++) {
					int cI_post = (cI + 1) % 3;

					double rightCoef = dE_div_dsigma[cI] + dE_div_dsigma[cI_post];
					double sum_sigma = sigma[cI] + sigma[cI_post];
					const double eps = 1.0e-6;
					if (sum_sigma < eps) {
						rightCoef /= 2.0 * eps;
					}
					else {
						rightCoef /= 2.0 * sum_sigma;
					}

					const double& leftCoef = BLeftCoef[cI];
					B[cI](0, 0) = B[cI](1, 1) = leftCoef + rightCoef;
					B[cI](0, 1) = B[cI](1, 0) = leftCoef - rightCoef;
					//if (projectSPD) {
					//	IglUtils::makePD2d(B[cI]);
					//}
				}


				// compute M using A(d2E_div_dsigma2) and B
				Eigen::Matrix<double, 9, 9> M;
				M.setZero();
				// A
				M(0, 0) = JW * d2E_div_dsigma2(0, 0);
				M(0, 4) = JW * d2E_div_dsigma2(0, 1);
				M(0, 8) = JW * d2E_div_dsigma2(0, 2);
				M(4, 0) = JW * d2E_div_dsigma2(1, 0);
				M(4, 4) = JW * d2E_div_dsigma2(1, 1);
				M(4, 8) = JW * d2E_div_dsigma2(1, 2);
				M(8, 0) = JW * d2E_div_dsigma2(2, 0);
				M(8, 4) = JW * d2E_div_dsigma2(2, 1);
				M(8, 8) = JW * d2E_div_dsigma2(2, 2);
				// B01
				M(1, 1) = JW * B[0](0, 0);
				M(1, 3) = JW * B[0](0, 1);
				M(3, 1) = JW * B[0](1, 0);
				M(3, 3) = JW * B[0](1, 1);
				// B12
				M(5, 5) = JW * B[1](0, 0);
				M(5, 7) = JW * B[1](0, 1);
				M(7, 5) = JW * B[1](1, 0);
				M(7, 7) = JW * B[1](1, 1);
				// B20
				M(2, 2) = JW * B[2](1, 1);
				M(2, 6) = JW * B[2](1, 0);
				M(6, 2) = JW * B[2](0, 1);
				M(6, 6) = JW * B[2](0, 0);


				// compute dP_div_dF
				Eigen::Matrix<double, 9, 9> wdP_div_dF;
				int dim = 3;
				for (int i = 0; i < dim; i++) {
					int _dim_i = i * dim;
					for (int j = 0; j < dim; j++) {
						int ij = _dim_i + j;
						for (int r = 0; r < dim; r++) {
							int _dim_r = r * dim;
							for (int s = 0; s < dim; s++) {
								int rs = _dim_r + s;
								if (ij > rs) {
									// bottom left, same as upper right
									continue;
								}

								wdP_div_dF(ij, rs) = 
									M(0, 0) * U(i, 0) * V(j, 0) * U(r, 0) * V(s, 0) + 
									M(0, 4) * U(i, 0) * V(j, 0) * U(r, 1) * V(s, 1) + 
									M(0, 8) * U(i, 0) * V(j, 0) * U(r, 2) * V(s, 2) + 
									M(4, 0) * U(i, 1) * V(j, 1) * U(r, 0) * V(s, 0) + 
									M(4, 4) * U(i, 1) * V(j, 1) * U(r, 1) * V(s, 1) + 
									M(4, 8) * U(i, 1) * V(j, 1) * U(r, 2) * V(s, 2) + 
									M(8, 0) * U(i, 2) * V(j, 2) * U(r, 0) * V(s, 0) + 
									M(8, 4) * U(i, 2) * V(j, 2) * U(r, 1) * V(s, 1) + 
									M(8, 8) * U(i, 2) * V(j, 2) * U(r, 2) * V(s, 2) + 
									M(1, 1) * U(i, 0) * V(j, 1) * U(r, 0) * V(s, 1) + 
									M(1, 3) * U(i, 0) * V(j, 1) * U(r, 1) * V(s, 0) + 
									M(3, 1) * U(i, 1) * V(j, 0) * U(r, 0) * V(s, 1) + 
									M(3, 3) * U(i, 1) * V(j, 0) * U(r, 1) * V(s, 0) + 
									M(5, 5) * U(i, 1) * V(j, 2) * U(r, 1) * V(s, 2) + 
									M(5, 7) * U(i, 1) * V(j, 2) * U(r, 2) * V(s, 1) + 
									M(7, 5) * U(i, 2) * V(j, 1) * U(r, 1) * V(s, 2) + 
									M(7, 7) * U(i, 2) * V(j, 1) * U(r, 2) * V(s, 1) + 
									M(2, 2) * U(i, 0) * V(j, 2) * U(r, 0) * V(s, 2) + 
									M(2, 6) * U(i, 0) * V(j, 2) * U(r, 2) * V(s, 0) + 
									M(6, 2) * U(i, 2) * V(j, 0) * U(r, 0) * V(s, 2) + 
									M(6, 6) * U(i, 2) * V(j, 0) * U(r, 2) * V(s, 0);

								if (ij < rs) {
									wdP_div_dF(rs, ij) = wdP_div_dF(ij, rs);
								}
							}
						}
					}
				}

				Eigen::Matrix<double, 3, 3>& A = restTriInv;


				//std::cout << J << " " << sigma.prod() << std::endl;

				//IglUtils::computeCofactorMtr(F, FInvT);
				//FInvT /= J;
				// 제1 Piola-Kirchhoff 응력 텐서 P = ∂Ψ / ∂F = μ F + (λ log(J) - μ) F ^ (-T) = μ(F-F^{-T}) + λ log(J) F ^ (-T)
				Eigen::Matrix<double, 3, 3> FInvT = F.inverse().transpose();
				Eigen::Matrix<double, 3, 3> dE_div_dF = mu_ * (F - FInvT) + lambda_ * logJ * FInvT;

				Eigen::Vector<double, 12> wdE_div_dx;
				wdE_div_dx[3] = A.row(0).dot(dE_div_dF.row(0));
				wdE_div_dx[4] = A.row(0).dot(dE_div_dF.row(1));
				wdE_div_dx[5] = A.row(0).dot(dE_div_dF.row(2));
				wdE_div_dx[6] = A.row(1).dot(dE_div_dF.row(0));
				wdE_div_dx[7] = A.row(1).dot(dE_div_dF.row(1));
				wdE_div_dx[8] = A.row(1).dot(dE_div_dF.row(2));
				wdE_div_dx[9] = A.row(2).dot(dE_div_dF.row(0));
				wdE_div_dx[10] = A.row(2).dot(dE_div_dF.row(1));
				wdE_div_dx[11] = A.row(2).dot(dE_div_dF.row(2));
				wdE_div_dx[0] = -wdE_div_dx[3] - wdE_div_dx[6] - wdE_div_dx[9];
				wdE_div_dx[1] = -wdE_div_dx[4] - wdE_div_dx[7] - wdE_div_dx[10];
				wdE_div_dx[2] = -wdE_div_dx[5] - wdE_div_dx[8] - wdE_div_dx[11];

				wdE_div_dx *= JW;



				Eigen::Matrix<double, 12, 9> wdP_div_dx;
				Eigen::Matrix<double, 12, 12> hessian_tmp;
				auto wdP_div_dF_transpose = wdP_div_dF.transpose();
				for (int colI = 0; colI < wdP_div_dF_transpose.cols(); colI++) {
					wdP_div_dx(3, colI) = (A.row(0) * wdP_div_dF_transpose.block(0, colI, dim, 1))[0];
					wdP_div_dx(4, colI) = (A.row(0) * wdP_div_dF_transpose.block(dim, colI, dim, 1))[0];
					wdP_div_dx(5, colI) = (A.row(0) * wdP_div_dF_transpose.block(dim * 2, colI, dim, 1))[0];
					wdP_div_dx(6, colI) = (A.row(1) * wdP_div_dF_transpose.block(0, colI, dim, 1))[0];
					wdP_div_dx(7, colI) = (A.row(1) * wdP_div_dF_transpose.block(dim, colI, dim, 1))[0];
					wdP_div_dx(8, colI) = (A.row(1) * wdP_div_dF_transpose.block(dim * 2, colI, dim, 1))[0];
					wdP_div_dx(9, colI) = (A.row(2) * wdP_div_dF_transpose.block(0, colI, dim, 1))[0];
					wdP_div_dx(10, colI) = (A.row(2) * wdP_div_dF_transpose.block(dim, colI, dim, 1))[0];
					wdP_div_dx(11, colI) = (A.row(2) * wdP_div_dF_transpose.block(dim * 2, colI, dim, 1))[0];
					wdP_div_dx(0, colI) = -wdP_div_dx(3, colI) - wdP_div_dx(6, colI) - wdP_div_dx(9, colI);
					wdP_div_dx(1, colI) = -wdP_div_dx(4, colI) - wdP_div_dx(7, colI) - wdP_div_dx(10, colI);
					wdP_div_dx(2, colI) = -wdP_div_dx(5, colI) - wdP_div_dx(8, colI) - wdP_div_dx(11, colI);
				}
				auto wdP_div_dx_transpose = wdP_div_dx.transpose();
				for (int colI = 0; colI < wdP_div_dx_transpose.cols(); colI++) {
					hessian_tmp(3, colI) = (A.row(0) * wdP_div_dx_transpose.block(0, colI, dim, 1))[0];
					hessian_tmp(4, colI) = (A.row(0) * wdP_div_dx_transpose.block(dim, colI, dim, 1))[0];
					hessian_tmp(5, colI) = (A.row(0) * wdP_div_dx_transpose.block(dim * 2, colI, dim, 1))[0];
					hessian_tmp(6, colI) = (A.row(1) * wdP_div_dx_transpose.block(0, colI, dim, 1))[0];
					hessian_tmp(7, colI) = (A.row(1) * wdP_div_dx_transpose.block(dim, colI, dim, 1))[0];
					hessian_tmp(8, colI) = (A.row(1) * wdP_div_dx_transpose.block(dim * 2, colI, dim, 1))[0];
					hessian_tmp(9, colI) = (A.row(2) * wdP_div_dx_transpose.block(0, colI, dim, 1))[0];
					hessian_tmp(10, colI) = (A.row(2) * wdP_div_dx_transpose.block(dim, colI, dim, 1))[0];
					hessian_tmp(11, colI) = (A.row(2) * wdP_div_dx_transpose.block(dim * 2, colI, dim, 1))[0];
					hessian_tmp(0, colI) = -hessian_tmp(3, colI) - hessian_tmp(6, colI) - hessian_tmp(9, colI);
					hessian_tmp(1, colI) = -hessian_tmp(4, colI) - hessian_tmp(7, colI) - hessian_tmp(10, colI);
					hessian_tmp(2, colI) = -hessian_tmp(5, colI) - hessian_tmp(8, colI) - hessian_tmp(11, colI);
				}



				for (int anode = 0; anode < 4; ++anode) {
					int iva = ivs[anode];

					for (int idim = 0; idim < 3; ++idim) {
						// gradient
						gradient(iva * dim + idim) += wdE_div_dx[anode * dim + idim];
					}

					for (int bnode = 0; bnode < 4; ++bnode) {
						int ivb = ivs[bnode];
						for (int idim = 0; idim < 3; ++idim) {
							for (int jdim = 0; jdim < 3; ++jdim) {
								// Hessian
								hessian(iva * dim + idim, ivb * dim + jdim) += 
									hessian_tmp(anode * dim + idim, bnode * dim + jdim);
							}

						}
					}
				}






				//=================================================
				// inertia energy
				// E = 1/(2Δ𝑡^{2}) (x − ˜x)^{T} M (x − ˜x)
				// 형상함수로 차분화
				// E = 1/(2Δ𝑡^{2}) (Δx_{a})^{T} N_{a}^{T} M N_{b} (Δx_{b})
				// 
				// 1차 미분 (gradient) :
				// dE/d(x_{a}) = 1/(2Δ𝑡^{2}) (Δx_{a})^{T} (N_{a}/x_{a})^{T} M N_{b} (Δx_{b})
				// 
				// 2차 미분 (hessian) : diagonal 항만 존재
				// dE^{2}/d^{2}(x_{a}) = 1/(Δ𝑡^{2}) (Δx_{a})^{T} (N_{a}/x_{a})^{T} M (N_{a}/x_{a}) (Δx_{a})
				// 
				for (int anode = 0; anode < 4; ++anode) {
					const auto& N_a = N_i[anode];
					int iva = ivs[anode];
					vec3 x_tilde_a = x0local.col(anode) + dt * vlocal.col(anode);// +dt * dt * a_q;
					for (int bnode = 0; bnode < 4; ++bnode) {
						const auto& N_b = N_i[bnode];
						int ivb = ivs[bnode];
						vec3 x_tilde_b = x0local.col(bnode) + dt * vlocal.col(bnode);
						double mass = density * N_a * N_b * JW;

						//for (int idim = 0; idim < 3; ++idim) {

						//	// gradient : inertia energy
						//	gradient(iva * 3 + idim) +=
						//		(1.0 / (2.0 * dt * dt) * mass) *
						//		(DN_x(idim, anode) * N_i[bnode] + DN_x(idim, bnode) * N_i[anode]) *
						//		(xlocal(idim, anode) - x_tilde_a[idim]) *
						//		(xlocal(idim, bnode) - x_tilde_b[idim]);

						//	// Hessian : inertia energy
						//	hessian(iva * 3 + idim, ivb * 3 + idim) +=
						//		(1.0 / (dt * dt) * mass) *
						//		DN_x(idim, anode) * DN_x(idim, bnode) *
						//		(xlocal(idim, anode) - x_tilde_a[idim]) *
						//		(xlocal(idim, bnode) - x_tilde_b[idim]);
						//}


					}
				}



				////=================================================
				//// elasticity potential 제약
				//// Neo-Hookean : Ψ_Neo=μ/2 (I_C-3)-μ log⁡J+λ/2 (log⁡J )^2
				//// 
				//// gradient : elasticity potential 
				//double psi =
				//	0.5 * mu_ * (Ib - 3.0) - mu_ * logJ +
				//	0.5 * lambda_ * logJ * logJ;
				//// 제1 Piola-Kirchhoff 응력 텐서 P = ∂Ψ / ∂F = μ F + (λ log(J) - μ) F ^ (-T)
				//Eigen::MatrixXd dpsidx = (mu_ * F + (lambda_ * logJ - mu_) * F.inverse().transpose()) * DN_x0;// 3 x 4
				//for (int anode = 0; anode < 4; ++anode) {
				//	const auto& N_a = N_i[anode];
				//	int iva = ivs[anode];
				//	// gradient 
				//	for (int idim = 0; idim < 3; ++idim) {
				//		gradient(iva * 3 + idim) += JW * dpsidx(idim, anode);


				//		// Hessian : 간단한 버젼
				//		hessian(iva * 3 + idim, iva * 3 + idim) += JW * (lambda_ + 2.0 * mu_) * DN_x0(idim, anode);
				//	}

				//	//for (int bnode = 0; bnode < 4; ++bnode) {
				//	//	const auto& N_b = N_i[bnode];
				//	//	int ivb = ivs[bnode];
				//	//	// Hessian 
				//	//	for (int idim = 0; idim < 3; ++idim) {
				//	//		hessian0(iva * 3 + idim, ivb * 3 + idim) += 0.0;
				//	//	}

				//	//}
				//}

				//// Hessian : elasticity potential 
				//// ∂²Ψ/∂x_i²= μ \Sigma_j(∂F_ij/∂x_i)^{2} +
				////				   λ(∂J/∂x_i)^{2} * 1/J^{2} +
				////				   (μ + λ log(J))(∂²J/∂x_i²)/J -
				////				   μ(∂²J/∂x_i²)/J
				//// 
				////Eigen::MatrixXd hessian_phi_NH_q =
				////	lambda_ * (F_invT.cwiseProduct(F_invT)) - 
				////	(lambda_ * logJ - mu_) * F_invT.cwiseProduct(F.inverse()) +
				////	mu_ * Eigen::Matrix3d::Identity();
				////hessian_phi_NH_q = DN_x0 * hessian_phi_NH_q * DN_x0.transpose();
				////hessian_phi_NH += hessian_phi_NH_q;

			}



		}

		return;

	};






	//=================================================
	// 거리 제약 (IPC [2])
	// E_c = -k_c * (d-d_hat)^2*ln(d/d_hat)
	// dE_c/dd = -k_c * 2 (1-d_hat) * ln(d/d_hat) -k_c * (d-d_hat)^2 * 1/d
	// d = sqrt((x_ngb-x_own)^2+(y_ngb-y_own)^2+(z_ngb-z_own)^2)
	// dd/dx_own = 0.5 (d)^{-0.5} * (-2.0(x_ngb-x_own))
	// dd/dx_ngb = 0.5 (d)^{-0.5} * ( 2.0(x_ngb-x_own))
	//double potential = 0.0;
	//if (d <= d_hat)
	//	potential = -k_c * std::pow(d - d_hat, 2.0) * std::log(d / d_hat);
	// 

	double d_hat = 1.6;
	double k_c = 0.1;

	auto contact_const = [k_c, d_hat](const vec3& p0, const vec3& v1) {
		using ADouble = TinyAD::Scalar<3, double, true>;
		Eigen::Vector<ADouble, 3> x_AD = ADouble::make_active(p0);
		ADouble dist = (x_AD - v1).squaredNorm();
		ADouble potential = -k_c * pow(dist - d_hat, 2.0) * log(dist / d_hat);
		return potential;
		};


	// 위치 랜덤하게 바꾸기
	// 난수 생성 엔진 초기화
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (int i = 0; i < mesh0.pos.rows(); ++i) {
		mesh0.pos.row(i)[0] = dist(gen);
		mesh0.pos.row(i)[1] = dist(gen);
		mesh0.pos.row(i)[2] = dist(gen);
	}
	for (int i = 0; i < mesh1.pos.rows(); ++i) {
		mesh1.pos.row(i)[0] = dist(gen);
		mesh1.pos.row(i)[1] = dist(gen);
		mesh1.pos.row(i)[2] = dist(gen);
	}




	for (int iter_glo = 0; iter_glo < 100; ++iter_glo) {

		hessian0.setZero(); hessian0.diagonal().setConstant(1.e-12);
		hessian1.setZero(); hessian1.diagonal().setConstant(1.e-12);
		gradient0.setZero();
		gradient1.setZero();

		inertia_elasticity_const(mesh0, gradient0, hessian0);
		inertia_elasticity_const(mesh1, gradient1, hessian1);

		//std::cout << hessian0 << std::endl;


		//// edge-edge
		//std::vector<std::tuple<double, vec3, vec3>> mesh0_min_dist_e(mesh0.e2v.rows(), std::tuple<double, vec3, vec3>{1.e100, {}, {}});
		//std::vector<std::tuple<double, vec3, vec3>> mesh1_min_dist_e(mesh1.e2v.rows(), std::tuple<double, vec3, vec3>{1.e100, {}, {}});
		//for (int i = 0; i < mesh0.e2v.rows(); ++i) {
		//	auto e0 = mesh0.e2v.row(i);
		//	Segment seg0, seg1;
		//	seg0.a = mesh0.pos.row(e0[0]);
		//	seg0.b = mesh0.pos.row(e0[1]);
		//	for (int j = 0; j < mesh1.e2v.rows(); ++j) {
		//		auto e1 = mesh1.e2v.row(j);
		//		seg1.a = mesh1.pos.row(e1[0]);
		//		seg1.b = mesh1.pos.row(e1[1]);
		//		auto [d, v0, v1] = closest_distance(seg0, seg1);

		//		//std::cout << d << std::endl;

		//		auto& [da, v0a, v1a] = mesh0_min_dist_e[i];
		//		auto& [db, v0b, v1b] = mesh1_min_dist_e[j];
		//		if (d < da) {
		//			da = d;	v0a = v0; v1a = v1;
		//		}
		//		if (d < db) {
		//			db = d;	v0b = v1; v1b = v0;
		//		}
		//	}
		//}


		//// point-triangle
		//std::vector<std::tuple<double, vec3, vec3>> mesh0_min_dist_f(mesh0.f2v.rows(), std::tuple<double, vec3, vec3>{1.e100, {}, {}});
		//std::vector<std::tuple<double, vec3, vec3>> mesh1_min_dist_f(mesh1.f2v.rows(), std::tuple<double, vec3, vec3>{1.e100, {}, {}});
		//std::vector<std::tuple<double, vec3, vec3>> mesh0_min_dist_p(mesh0.pos.rows(), std::tuple<double, vec3, vec3>{1.e100, {}, {}});
		//std::vector<std::tuple<double, vec3, vec3>> mesh1_min_dist_p(mesh1.pos.rows(), std::tuple<double, vec3, vec3>{1.e100, {}, {}});
		//for (int i = 0; i < mesh0.f2v.rows(); ++i) {
		//	auto f0 = mesh0.f2v.row(i);
		//	Triangle tri;
		//	Point p;
		//	tri.a = mesh0.pos.row(f0[0]);
		//	tri.b = mesh0.pos.row(f0[1]);
		//	tri.c = mesh0.pos.row(f0[2]);
		//	for (int j = 0; j < mesh1.pos.rows(); ++j) {
		//		p = mesh1.pos.row(j);
		//		auto [d, v0, v1] = closest_distance(p, tri);

		//		auto& [da, v0a, v1a] = mesh0_min_dist_f[i];
		//		auto& [db, v0b, v1b] = mesh1_min_dist_p[j];
		//		if (d < da) {
		//			da = d;	v0a = v1; v1a = v0;
		//		}
		//		if (d < db) {
		//			db = d;	v0b = v0; v1b = v1;
		//		}

		//	}
		//}
		//for (int i = 0; i < mesh1.f2v.rows(); ++i) {
		//	auto f0 = mesh1.f2v.row(i);
		//	Triangle tri;
		//	Point p;
		//	tri.a = mesh1.pos.row(f0[0]);
		//	tri.b = mesh1.pos.row(f0[1]);
		//	tri.c = mesh1.pos.row(f0[2]);
		//	for (int j = 0; j < mesh0.pos.rows(); ++j) {
		//		p = mesh0.pos.row(j);
		//		auto [d, v0, v1] = closest_distance(p, tri);

		//		auto& [da, v0a, v1a] = mesh1_min_dist_f[i];
		//		auto& [db, v0b, v1b] = mesh0_min_dist_p[j];
		//		if (d < da) {
		//			da = d;	v0a = v1; v1a = v0;
		//		}
		//		if (d < db) {
		//			db = d;	v0b = v0; v1b = v1;
		//		}

		//	}
		//}



		//// edge
		//for (int i = 0; i < mesh0_min_dist_e.size(); ++i) {
		//	auto& [d, v0, v1] = mesh0_min_dist_e[i];
		//	assert(d > 0.0);
		//	auto iv0 = mesh0.e2v(i, 0);
		//	auto iv1 = mesh0.e2v(i, 1);
		//	vec3 p0 = mesh0.pos.row(iv0);
		//	vec3 p1 = mesh0.pos.row(iv1);
		//	auto potential0 = contact_const(p0, v1);
		//	auto potential1 = contact_const(p1, v1);

		//	vec3 dddx_own = 0.5 * std::pow(d, -0.5) * (-2.0 * (v1 - v0));
		//	//vec3 dddx_ngb = 0.5 * std::pow(d, -0.5) * ( 2.0 * (v1 - v0));

		//	vec3 psi_grad = k_c * barrier_first_derivative(d, d_hat) * dddx_own;
		//	double psi_hess = std::pow(d, -0.5) * k_c * barrier_second_derivative(d, d_hat);

		//	gradient0.segment<3>(iv0 * 3) += psi_grad;
		//	gradient0.segment<3>(iv1 * 3) += psi_grad;

		//	hessian0(iv0 * 3 + 0, iv0 * 3 + 0) += psi_hess;
		//	hessian0(iv0 * 3 + 1, iv0 * 3 + 1) += psi_hess;
		//	hessian0(iv0 * 3 + 2, iv0 * 3 + 2) += psi_hess;
		//	hessian0(iv1 * 3 + 0, iv1 * 3 + 0) += psi_hess;
		//	hessian0(iv1 * 3 + 1, iv1 * 3 + 1) += psi_hess;
		//	hessian0(iv1 * 3 + 2, iv1 * 3 + 2) += psi_hess;

		//	//gradient0.segment<3>(iv0 * 3) += potential0.grad;
		//	//hessian0.block<3, 3>(iv0 * 3, iv0 * 3) += potential0.Hess;
		//	//gradient0.segment<3>(iv1 * 3) += potential1.grad;
		//	//hessian0.block<3, 3>(iv1 * 3, iv1 * 3) += potential1.Hess;

		//}
		//for (int i = 0; i < mesh1_min_dist_e.size(); ++i) {
		//	auto& [d, v0, v1] = mesh1_min_dist_e[i];
		//	assert(d > 0.0);
		//	auto iv0 = mesh1.e2v(i, 0);
		//	auto iv1 = mesh1.e2v(i, 1);
		//	vec3 p0 = mesh1.pos.row(iv0);
		//	vec3 p1 = mesh1.pos.row(iv1);
		//	auto potential0 = contact_const(p0, v1);
		//	auto potential1 = contact_const(p1, v1);

		//	vec3 dddx_own = 0.5 * std::pow(d, -0.5) * (-2.0 * (v1 - v0));
		//	//vec3 dddx_ngb = 0.5 * std::pow(d, -0.5) * ( 2.0 * (v1 - v0));

		//	vec3 psi_grad = k_c * barrier_first_derivative(d, d_hat) * dddx_own;
		//	double psi_hess = std::pow(d, -0.5) * k_c * barrier_second_derivative(d, d_hat);

		//	gradient1.segment<3>(iv0 * 3) += psi_grad;
		//	gradient1.segment<3>(iv1 * 3) += psi_grad;

		//	hessian1(iv0 * 3 + 0, iv0 * 3 + 0) += psi_hess;
		//	hessian1(iv0 * 3 + 1, iv0 * 3 + 1) += psi_hess;
		//	hessian1(iv0 * 3 + 2, iv0 * 3 + 2) += psi_hess;
		//	hessian1(iv1 * 3 + 0, iv1 * 3 + 0) += psi_hess;
		//	hessian1(iv1 * 3 + 1, iv1 * 3 + 1) += psi_hess;
		//	hessian1(iv1 * 3 + 2, iv1 * 3 + 2) += psi_hess;


		//	//gradient1.segment<3>(iv0 * 3) += potential0.grad;
		//	//hessian1.block<3, 3>(iv0 * 3, iv0 * 3) += potential0.Hess;
		//	//gradient1.segment<3>(iv1 * 3) += potential1.grad;
		//	//hessian1.block<3, 3>(iv1 * 3, iv1 * 3) += potential1.Hess;
		//}
		//// tri
		//for (int i = 0; i < mesh0_min_dist_f.size(); ++i) {
		//	auto& [d, v0, v1] = mesh0_min_dist_f[i];
		//	assert(d > 0.0);
		//	auto iv0 = mesh0.f2v(i, 0);
		//	auto iv1 = mesh0.f2v(i, 1);
		//	auto iv2 = mesh0.f2v(i, 2);
		//	vec3 p0 = mesh0.pos.row(iv0);
		//	vec3 p1 = mesh0.pos.row(iv1);
		//	vec3 p2 = mesh0.pos.row(iv2);
		//	auto potential0 = contact_const(p0, v1);
		//	auto potential1 = contact_const(p1, v1);
		//	auto potential2 = contact_const(p2, v1);

		//	vec3 dddx_own = 0.5 * std::pow(d, -0.5) * (-2.0 * (v1 - v0));
		//	//vec3 dddx_ngb = 0.5 * std::pow(d, -0.5) * ( 2.0 * (v1 - v0));

		//	vec3 psi_grad = k_c * barrier_first_derivative(d, d_hat) * dddx_own;
		//	double psi_hess = std::pow(d, -0.5) * k_c * barrier_second_derivative(d, d_hat);

		//	gradient0.segment<3>(iv0 * 3) += psi_grad;
		//	gradient0.segment<3>(iv1 * 3) += psi_grad;
		//	gradient0.segment<3>(iv2 * 3) += psi_grad;

		//	hessian0(iv0 * 3 + 0, iv0 * 3 + 0) += psi_hess;
		//	hessian0(iv0 * 3 + 1, iv0 * 3 + 1) += psi_hess;
		//	hessian0(iv0 * 3 + 2, iv0 * 3 + 2) += psi_hess;
		//	hessian0(iv1 * 3 + 0, iv1 * 3 + 0) += psi_hess;
		//	hessian0(iv1 * 3 + 1, iv1 * 3 + 1) += psi_hess;
		//	hessian0(iv1 * 3 + 2, iv1 * 3 + 2) += psi_hess;
		//	hessian0(iv2 * 3 + 0, iv2 * 3 + 0) += psi_hess;
		//	hessian0(iv2 * 3 + 1, iv2 * 3 + 1) += psi_hess;
		//	hessian0(iv2 * 3 + 2, iv2 * 3 + 2) += psi_hess;

		//	//gradient0.segment<3>(iv0 * 3) += potential0.grad;
		//	//hessian0.block<3, 3>(iv0 * 3, iv0 * 3) += potential0.Hess;
		//	//gradient0.segment<3>(iv1 * 3) += potential1.grad;
		//	//hessian0.block<3, 3>(iv1 * 3, iv1 * 3) += potential1.Hess;
		//	//gradient0.segment<3>(iv2 * 3) += potential2.grad;
		//	//hessian0.block<3, 3>(iv2 * 3, iv2 * 3) += potential2.Hess;
		//}
		//for (int i = 0; i < mesh1_min_dist_f.size(); ++i) {
		//	auto& [d, v0, v1] = mesh1_min_dist_f[i];
		//	assert(d > 0.0);
		//	auto iv0 = mesh1.f2v(i, 0);
		//	auto iv1 = mesh1.f2v(i, 1);
		//	auto iv2 = mesh1.f2v(i, 2);
		//	vec3 p0 = mesh1.pos.row(iv0);
		//	vec3 p1 = mesh1.pos.row(iv1);
		//	vec3 p2 = mesh1.pos.row(iv2);
		//	auto potential0 = contact_const(p0, v1);
		//	auto potential1 = contact_const(p1, v1);
		//	auto potential2 = contact_const(p2, v1);


		//	vec3 dddx_own = 0.5 * std::pow(d, -0.5) * (-2.0 * (v1 - v0));
		//	//vec3 dddx_ngb = 0.5 * std::pow(d, -0.5) * ( 2.0 * (v1 - v0));

		//	vec3 psi_grad = k_c * barrier_first_derivative(d, d_hat) * dddx_own;
		//	double psi_hess = std::pow(d, -0.5) * k_c * barrier_second_derivative(d, d_hat);


		//	gradient1.segment<3>(iv0 * 3) += psi_grad;
		//	gradient1.segment<3>(iv1 * 3) += psi_grad;
		//	gradient1.segment<3>(iv2 * 3) += psi_grad;

		//	hessian1(iv0 * 3 + 0, iv0 * 3 + 0) += psi_hess;
		//	hessian1(iv0 * 3 + 1, iv0 * 3 + 1) += psi_hess;
		//	hessian1(iv0 * 3 + 2, iv0 * 3 + 2) += psi_hess;
		//	hessian1(iv1 * 3 + 0, iv1 * 3 + 0) += psi_hess;
		//	hessian1(iv1 * 3 + 1, iv1 * 3 + 1) += psi_hess;
		//	hessian1(iv1 * 3 + 2, iv1 * 3 + 2) += psi_hess;
		//	hessian1(iv2 * 3 + 0, iv2 * 3 + 0) += psi_hess;
		//	hessian1(iv2 * 3 + 1, iv2 * 3 + 1) += psi_hess;
		//	hessian1(iv2 * 3 + 2, iv2 * 3 + 2) += psi_hess;

		//	//gradient1.segment<3>(iv0 * 3) += potential0.grad;
		//	//hessian1.block<3, 3>(iv0 * 3, iv0 * 3) += potential0.Hess;
		//	//gradient1.segment<3>(iv1 * 3) += potential1.grad;
		//	//hessian1.block<3, 3>(iv1 * 3, iv1 * 3) += potential1.Hess;
		//	//gradient1.segment<3>(iv2 * 3) += potential2.grad;
		//	//hessian1.block<3, 3>(iv2 * 3, iv2 * 3) += potential2.Hess;
		//}
		//// point
		//for (int i = 0; i < mesh0_min_dist_p.size(); ++i) {
		//	auto& [d, v0, v1] = mesh0_min_dist_p[i];
		//	assert(d > 0.0);
		//	vec3 p0 = mesh0.pos.row(i);
		//	auto potential0 = contact_const(p0, v1);

		//	vec3 dddx_own = 0.5 * std::pow(d, -0.5) * (-2.0 * (v1 - v0));
		//	//vec3 dddx_ngb = 0.5 * std::pow(d, -0.5) * ( 2.0 * (v1 - v0));

		//	vec3 psi_grad = k_c * barrier_first_derivative(d, d_hat) * dddx_own;
		//	double psi_hess = std::pow(d, -0.5) * k_c * barrier_second_derivative(d, d_hat);

		//	gradient0.segment<3>(i * 3) += psi_grad;
		//	hessian0(i * 3 + 0, i * 3 + 0) += psi_hess;

		//	//gradient0.segment<3>(i * 3) += potential0.grad;
		//	//hessian0.block<3, 3>(i * 3, i * 3) += potential0.Hess;
		//}
		//for (int i = 0; i < mesh1_min_dist_p.size(); ++i) {
		//	auto& [d, v0, v1] = mesh1_min_dist_p[i];
		//	assert(d > 0.0);
		//	vec3 p0 = mesh1.pos.row(i);
		//	auto potential0 = contact_const(p0, v1);

		//	vec3 dddx_own = 0.5 * std::pow(d, -0.5) * (-2.0 * (v1 - v0));
		//	//vec3 dddx_ngb = 0.5 * std::pow(d, -0.5) * ( 2.0 * (v1 - v0));

		//	vec3 psi_grad = k_c * barrier_first_derivative(d, d_hat) * dddx_own;
		//	double psi_hess = std::pow(d, -0.5) * k_c * barrier_second_derivative(d, d_hat);

		//	gradient1.segment<3>(i * 3) += psi_grad;
		//	hessian1(i * 3 + 0, i * 3 + 0) += psi_hess;

		//	//gradient1.segment<3>(i * 3) += potential0.grad;
		//	//hessian1.block<3, 3>(i * 3, i * 3) += potential0.Hess;
		//}


		Eigen::VectorXd du0 = -hessian0.inverse() * gradient0;
		Eigen::VectorXd du1 = -hessian1.inverse() * gradient1;

		std::cout << du0.norm() << " " << du1.norm() << std::endl;


		for (int i = 0; i < mesh0.pos.rows(); ++i) {
			mesh0.pos.row(i)[0] += 1.e-1 * du0[3 * i + 0];
			mesh0.pos.row(i)[1] += 1.e-1 * du0[3 * i + 1];
			mesh0.pos.row(i)[2] += 1.e-1 * du0[3 * i + 2];
		}
		for (int i = 0; i < mesh1.pos.rows(); ++i) {
			mesh1.pos.row(i)[0] += 1.e-1 * du1[3 * i + 0];
			mesh1.pos.row(i)[1] += 1.e-1 * du1[3 * i + 1];
			mesh1.pos.row(i)[2] += 1.e-1 * du1[3 * i + 2];
		}

	}


	make_vtu(mesh0, "mesh0_1.vtu");
	make_vtu(mesh1, "mesh1_1.vtu");





	return 0;
}

//
//int main() {
//
//
//    auto func = TinyAD::scalar_function<2>(TinyAD::range(1));
//    func.add_elements<2>(TinyAD::range(1), [](auto& element)->TINYAD_SCALAR_TYPE(element)
//    {
//        using T = TINYAD_SCALAR_TYPE(element);
//        Eigen::Vector<T, 2> v = element.variables(0);
//        return (v[0] - 1) * (v[0] - 1) + (v[1] - 2) * (v[1] - 2);
//    });
//
//
//    //const double f = func.eval(Eigen::Vector<double, 2>(1.0 , 1.0));
//    auto [f, g, H] = func.eval_with_derivatives(Eigen::Vector<double, 2>(1.0, 1.0));
//
//    std::cout << f << std::endl << std::endl;
//    std::cout << g << std::endl << std::endl;
//    std::cout << H << std::endl << std::endl;
//
//    // OSQP 문제 설정
//    OSQPInt n = 2;  // 변수 개수
//    OSQPInt m = 3;  // 제약 조건 개수
//
//    // Hessian 행렬 설정 (상삼각 부분만)
//    OSQPFloat P_x[3] = { H.coeff(0,0), H.coeff(0,1), H.coeff(1,1) };
//    OSQPInt P_nnz = 3;
//    OSQPInt P_i[3] = { 0, 0, 1 };
//    OSQPInt P_p[3] = { 0, 1, 3 };
//
//    // 선형항 설정
//    OSQPFloat q[2] = { g[0], g[1] };
//
//    // 제약 조건 설정 (예: x + y = 1, 0 <= x <= 0.7, 0 <= y <= 0.7)
//    OSQPFloat A_x[4] = { 1.0, 1.0, 1.0, 1.0 };
//    OSQPInt A_nnz = 4;
//    OSQPInt A_i[4] = { 0, 1, 0, 2 };
//    OSQPInt A_p[3] = { 0, 2, 4 };
//    OSQPFloat l[3] = { 1.0, 0.0, 0.0 };
//    OSQPFloat u[3] = { 1.0, 0.7, 0.7 };
//
//    /* Exitflag */
//    OSQPInt exitflag = 0;
//
//    /* Solver, settings, matrices */
//    OSQPSolver* solver = nullptr;
//    OSQPSettings* settings = nullptr;
//    OSQPCscMatrix* P = new OSQPCscMatrix;
//    OSQPCscMatrix* A = new OSQPCscMatrix;
//
//    /* Populate matrices */
//    csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
//    csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);
//
//    /* Set default settings */
//    settings = new OSQPSettings;
//    if (settings) {
//        osqp_set_default_settings(settings);
//        settings->alpha = 1.0; /* Change alpha parameter */
//    }
//
//    /* Setup solver */
//    exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);
//
//    /* Solve problem */
//    if (!exitflag) exitflag = osqp_solve(solver);
//
//    /* Print solution */
//    if (!exitflag) {
//        std::cout << "최적해: x = " << solver->solution->x[0]
//            << ", y = " << solver->solution->x[1] << std::endl;
//    }
//
//    /* Cleanup */
//    osqp_cleanup(solver);
//    delete A;
//    delete P;
//    delete settings;
//
//    //return static_cast<int>(exitflag);
//
//    return 0;
//
//
//	//// Choose autodiff scalar type for 3 variables
//	//using ADouble = TinyAD::Double<3>;
//
//	//// Init a 3D vector of active variables and a 3D vector of passive variables
//	//Eigen::Vector3<ADouble> x = ADouble::make_active({ 0.0, -1.0, 1.0 });
//	//Eigen::Vector3<double> y(2.0, 3.0, 5.0);
//
//	//// Compute angle using Eigen functions and retrieve gradient and Hessian w.r.t. x
//	//ADouble angle = acos(x.dot(y) / (x.norm() * y.norm()));
//	//Eigen::Vector3d g = angle.grad;
//	//Eigen::Matrix3d H = angle.Hess;
//
//	//std::cout << g.transpose() << std::endl;
//	//std::cout << H.transpose() << std::endl;
//
//
//	//// Set up a function with 2D vertex positions as variables
//	//auto func = TinyAD::scalar_function<2>();
//
//	//// Add an objective term per triangle. Each connecting 3 vertices
//	//func.add_elements<3>(mesh.faces(), [&](auto& element)
//	//	{
//	//		// Element is evaluated with either double or TinyAD::Double<6>
//	//		using T = TINYAD_SCALAR_TYPE(element);
//
//	//		// Get variable 2D vertex positions of triangle t
//	//		OpenMesh::SmartFaceHandle t = element.handle;
//	//		Eigen::Vector2<T> a = element.variables(t.halfedge().to());
//	//		Eigen::Vector2<T> b = element.variables(t.halfedge().next().to());
//	//		Eigen::Vector2<T> c = element.variables(t.halfedge().from());
//
//	//		return ...
//	//	});
//
//	//// Evaluate the funcion using any of these methods:
//	//double f = func.eval(x);
//	//auto [f, g] = func.eval_with_gradient(x);
//	//auto [f, g, H] = func.eval_with_derivatives(x);
//	//auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
//
//}
//



//#include <cstdlib>
//#include "osqp.h"
//
//int main(int argc, char** argv) {
//    /* Load problem data */
//    OSQPFloat P_x[3] = { 4.0, 1.0, 2.0 };
//    OSQPInt P_nnz = 3;
//    OSQPInt P_i[3] = { 0, 0, 1 };
//    OSQPInt P_p[3] = { 0, 1, 3 };
//    OSQPFloat q[2] = { 1.0, 1.0 };
//    OSQPFloat A_x[4] = { 1.0, 1.0, 1.0, 1.0 };
//    OSQPInt A_nnz = 4;
//    OSQPInt A_i[4] = { 0, 1, 0, 2 };
//    OSQPInt A_p[3] = { 0, 2, 4 };
//    OSQPFloat l[3] = { 1.0, 0.0, 0.0 };
//    OSQPFloat u[3] = { 1.0, 0.7, 0.7 };
//    OSQPInt n = 2;
//    OSQPInt m = 3;
//
//    /* Exitflag */
//    OSQPInt exitflag = 0;
//
//    /* Solver, settings, matrices */
//    OSQPSolver* solver = nullptr;
//    OSQPSettings* settings = nullptr;
//    OSQPCscMatrix* P = new OSQPCscMatrix;
//    OSQPCscMatrix* A = new OSQPCscMatrix;
//
//    /* Populate matrices */
//    csc_set_data(A, m, n, A_nnz, A_x, A_i, A_p);
//    csc_set_data(P, n, n, P_nnz, P_x, P_i, P_p);
//    
//    /* Set default settings */
//    settings = new OSQPSettings;
//    if (settings) {
//        osqp_set_default_settings(settings);
//        settings->alpha = 1.0; /* Change alpha parameter */
//    }
//
//    /* Setup solver */
//    exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);
//
//    /* Solve problem */
//    if (!exitflag) exitflag = osqp_solve(solver);
//
//    /* Cleanup */
//    osqp_cleanup(solver);
//    delete A;
//    delete P;
//    delete settings;
//
//    return static_cast<int>(exitflag);
//}
