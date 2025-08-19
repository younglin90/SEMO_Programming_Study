
#include <iostream>
#include <random>
#include <variant>

#include <Eigen/Dense>
#include <finite-diff/finitediff.hpp>
#include "./3rd-party/autodiff/reverse/var.hpp"
#include "./3rd-party/autodiff/reverse/var/eigen.hpp"



template<typename TYPE>
struct EOS_ideal {
	double gamma_, cv_, cp_;

	EOS_ideal() = default;

	EOS_ideal(double gamma, double cv) : gamma_(gamma), cv_(cv) {
		cp_ = gamma_ * cv_;
	}

	void operator()(
		const TYPE& p,
		const TYPE& u,
		const TYPE& v,
		const TYPE& w,
		const TYPE& T,
		TYPE& rho,
		TYPE& Ht,
		TYPE& drhodp,
		TYPE& drhodT,
		TYPE& dHtdp,
		TYPE& dHtdT,
		TYPE& drhodpdp,
		TYPE& drhodpdT,
		TYPE& drhodTdT
	) const {
		TYPE usqrt = u * u + v * v + w * w;

		// density of each phase
		rho = 1.0 / ((gamma_ - 1.0) * cv_ * T / p);

		// d(rho)/d(p)
		drhodp = rho / p;

		// d(rho)/d(T)
		drhodT = -rho / T;

		// d(h)/d(p)
		dHtdp = 0.0;
		// d(h)/d(T)
		dHtdT = gamma_ * cv_;

		// internal energy of each phase
		TYPE internal_energy = cv_ * T;

		// enthalpy of each phase
		TYPE enthalpy = gamma_ * cv_ * T;

		// eti = internal_energy + 0.5*usqrt
		TYPE Et = internal_energy + 0.5 * usqrt;
		Ht = enthalpy + 0.5 * usqrt;

		// 추가적인 정보
		drhodpdp = drhodp / p - rho / p / p;
		drhodpdT = drhodT / p;
		drhodTdT = -drhodT / T + rho / T / T;

	}
};


template<typename TYPE>
struct EOS_NASG {
	double gamma_, cv_, cp_;
	double pinf_, b_, q_;

	EOS_NASG() = default;

	EOS_NASG(double gamma, double cv, double pinf, double b, double q) :
		gamma_(gamma), cv_(cv),
		pinf_(pinf), b_(b),
		q_(q)
	{
		cp_ = gamma_ * cv_;
	}

	void operator()(
		const TYPE& p,
		const TYPE& u,
		const TYPE& v,
		const TYPE& w,
		const TYPE& T,
		TYPE& rho,
		TYPE& Ht,
		TYPE& drhodp,
		TYPE& drhodT,
		TYPE& dHtdp,
		TYPE& dHtdT,
		TYPE& drhodpdp,
		TYPE& drhodpdT,
		TYPE& drhodTdT
	) const {
		TYPE usqrt = u * u + v * v + w * w;

		// density of each phase
		rho = 1.0 / ((gamma_ - 1.0) * cv_ * T / (p + pinf_) + b_);

		// d(rho)/d(p)
		drhodp = rho * rho * (1.0 / rho - b_) / (p + pinf_);

		// d(rho)/d(T)
		drhodT = -rho * rho * (1.0 / rho - b_) / T;

		// d(h)/d(p)
		dHtdp = b_;
		// d(h)/d(T)
		dHtdT = gamma_ * cv_;

		// internal energy of each phase
		TYPE internal_energy = cv_ * T;

		// enthalpy of each phase
		TYPE enthalpy = gamma_ * cv_ * T + b_ * p + q_;

		// eti = internal_energy + 0.5*usqrt
		TYPE Et = internal_energy + 0.5 * usqrt;
		Ht = enthalpy + 0.5 * usqrt;

		// 추가적인 정보
		drhodpdp = (drhodp - 2.0 * rho * drhodp * b_) / (p + pinf_) - (rho - rho * rho * b_) / (p + pinf_) / (p + pinf_);
		drhodpdT = (drhodT - 2.0 * rho * drhodT * b_) / (p + pinf_);
		drhodTdT = -(drhodT - 2.0 * rho * drhodT * b_) / T + (rho - rho * rho * b_) / T / T;

	}
};




template<typename TYPE, typename EOS_VEC_T>
inline void mixture_thermodynamic_properties(
	int species_size, const EOS_VEC_T& func_EOSs,
	const TYPE& p,
	const TYPE& u,
	const TYPE& v,
	const TYPE& w,
	const TYPE& T,
	const std::vector<TYPE>& MF,
	TYPE& rho,
	TYPE& Ht,
	TYPE& c,
	TYPE& dHtdp,
	TYPE& dHtdT,
	TYPE& drhodp,
	TYPE& drhodT,
	TYPE& drhodpdp,
	TYPE& drhodpdT,
	TYPE& drhodTdT,
	std::vector<TYPE>& drhodMF,
	std::vector<TYPE>& dHtdMF,
	std::vector<TYPE>& drhodpdMF,
	std::vector<TYPE>& drhodTdMF,
	std::vector<TYPE>& dHtdpdMF,
	std::vector<TYPE>& dHtdTdMF
) {

	rho = 0.0, Ht = 0.0;
	dHtdp = 0.0;
	dHtdT = 0.0;
	drhodp = 0.0;
	drhodT = 0.0;
	TYPE drhodpdp_1 = 0.0;
	TYPE drhodpdp_2 = 0.0;
	TYPE drhodpdT_1 = 0.0;
	TYPE drhodpdT_2 = 0.0;
	TYPE drhodTdT_1 = 0.0;
	TYPE drhodTdT_2 = 0.0;
	std::vector<TYPE> rho_s(species_size);
	std::vector<TYPE> Ht_s(species_size);
	std::vector<TYPE> drhodp_s(species_size);
	std::vector<TYPE> drhodT_s(species_size);
	std::vector<TYPE> dHtdp_s(species_size);
	std::vector<TYPE> dHtdT_s(species_size);
	for (int isp = 0; isp < species_size; ++isp) {
		// EOS 계산
		TYPE drhodpdp_i, drhodpdT_i, drhodTdT_i;
		auto& rho_i = rho_s[isp];
		auto& Ht_i = Ht_s[isp];
		auto& drhodp_i = drhodp_s[isp];
		auto& drhodT_i = drhodT_s[isp];
		auto& dHtdp_i = dHtdp_s[isp];
		auto& dHtdT_i = dHtdT_s[isp];

		std::visit([&](auto&& calc_EOS) {
			return calc_EOS(p, u, v, w, T,
				rho_i, Ht_i, drhodp_i, drhodT_i, dHtdp_i, dHtdT_i, drhodpdp_i, drhodpdT_i, drhodTdT_i);
		}, func_EOSs[isp]);


		// 혼합물 물성치 계산 (mass fraction, Y 로 계산)
		const TYPE& Yi = MF[isp];
		rho += Yi / rho_i;
		Ht += Yi * Ht_i;
		dHtdp += Yi * dHtdp_i;
		dHtdT += Yi * dHtdT_i;
		drhodp += (Yi / rho_i / rho_i * drhodp_i);
		drhodT += (Yi / rho_i / rho_i * drhodT_i);

		drhodpdp_1 += Yi * drhodpdp_i / rho_i / rho_i;
		drhodpdp_2 += -2.0 * Yi / rho_i / rho_i / rho_i * drhodp_i * drhodp_i;

		drhodpdT_1 += Yi * drhodpdT_i / rho_i / rho_i;
		drhodpdT_2 += -2.0 * Yi / rho_i / rho_i / rho_i * drhodp_i * drhodT_i;

		drhodTdT_1 += Yi * drhodTdT_i / rho_i / rho_i;
		drhodTdT_2 += -2.0 * Yi / rho_i / rho_i / rho_i * drhodT_i * drhodT_i;

	}
	rho = 1.0 / rho;

	drhodpdp = rho * rho * (2.0 * rho * drhodp * drhodp + drhodpdp_1 + drhodpdp_2);
	drhodpdT = rho * rho * (2.0 * rho * drhodp * drhodT + drhodpdT_1 + drhodpdT_2);
	drhodTdT = rho * rho * (2.0 * rho * drhodT * drhodT + drhodTdT_1 + drhodTdT_2);

	drhodp *= rho * rho;
	drhodT *= rho * rho;

	for (int isp = 0; isp < species_size - 1; ++isp) {
		// 볼륨 프랙션 계산
		//VF[isp] = rho * MF[isp] / rho_s[isp];

		// 필요한 차분값 계산
		drhodMF[isp] = -rho * rho / rho_s[isp] + rho * rho / rho_s.back();
		dHtdMF[isp] = Ht_s[isp] - Ht_s.back();


		const TYPE rho_i = rho_s[isp];
		const TYPE rho_i_inv2 = 1.0 / (rho_i * rho_i);
		drhodpdMF[isp] =
			-2.0 * rho * drhodp / rho_i 
			+ rho * rho * drhodp_s[isp] * rho_i_inv2
			- (-2.0 * rho * drhodp / rho_s.back()
				+ rho * rho * drhodp_s.back()
				/ (rho_s.back() * rho_s.back()));
		drhodTdMF[isp] =
			-2.0 * rho * drhodT / rho_i
			+ rho * rho * drhodT_s[isp] * rho_i_inv2
			- (-2.0 * rho * drhodT / rho_s.back()
				+ rho * rho * drhodT_s.back()
				/ (rho_s.back() * rho_s.back()));
		dHtdpdMF[isp] = dHtdp_s[isp] - dHtdp_s.back();
		dHtdTdMF[isp] = dHtdT_s[isp] - dHtdT_s.back();
	}

	c = drhodp + drhodT / dHtdT * (1.0 / rho - dHtdp);
	c = sqrt(1.0 / c);

}




std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
calc_convective_flux_jacobian(
	int equation_size,
	int eq_str_spec_,
	int eq_size_spec_,
	const Eigen::Vector3d& nvec,
	const autodiff::MatrixXvar& phiFLR,
	const Eigen::MatrixXd& DWF_phiLR,
	const autodiff::var& drhoFLdpFL, const autodiff::var& drhoFRdpFR,
	const autodiff::var& drhoFLdTFL, const autodiff::var& drhoFRdTFR,
	const std::vector<autodiff::var>& drhoFLdMFFL,
	const std::vector<autodiff::var>& drhoFRdMFFR,
	const autodiff::var& dHtFLdpFL, const autodiff::var& dHtFRdpFR,
	const autodiff::var& dHtFLdTFL, const autodiff::var& dHtFRdTFR,
	const std::vector<autodiff::var>& dHtFLdMFFL, const std::vector<autodiff::var>& dHtFRdMFFR,
	const autodiff::var& cFL, const autodiff::var& cFR,
	const autodiff::var& rhoFL, const autodiff::var& rhoFR,
	const autodiff::var& HtFL, const autodiff::var& HtFR,
	const autodiff::var& drhoFLdpdpFL, const autodiff::var& drhoFRdpdpFR,
	const autodiff::var& drhoFLdpdTFL, const autodiff::var& drhoFRdpdTFR,
	const autodiff::var& drhoFLdTdTFL, const autodiff::var& drhoFRdTdTFR,
	const std::vector<autodiff::var>& dHtdpdMFFL, const std::vector<autodiff::var>& dHtdpdMFFR,
	const std::vector<autodiff::var>& dHtdTdMFFL, const std::vector<autodiff::var>& dHtdTdMFFR,
	const std::vector<autodiff::var>& drhoFLdpdMFFL, const std::vector<autodiff::var>& drhoFRdpdMFFR,
	const std::vector<autodiff::var>& drhoFLdTdMFFL, const std::vector<autodiff::var>& drhoFRdTdMFFR
) {



	//=======================================================

	Eigen::MatrixXd dphiFLdphiLR(2, equation_size);
	Eigen::MatrixXd dphiFRdphiLR(2, equation_size);
	dphiFLdphiLR.row(0) = (1.0 - DWF_phiLR.row(0).array());
	dphiFLdphiLR.row(1) = DWF_phiLR.row(0);
	dphiFRdphiLR.row(0) = DWF_phiLR.row(1);
	dphiFRdphiLR.row(1) = (1.0 - DWF_phiLR.row(1).array());

	Eigen::MatrixXd drhoFLdphiLR(2, equation_size);
	Eigen::MatrixXd drhoFRdphiLR(2, equation_size);
	drhoFLdphiLR(0, 0) = dphiFLdphiLR(0, 0) * val(drhoFLdpFL);
	drhoFLdphiLR(1, 0) = dphiFLdphiLR(1, 0) * val(drhoFLdpFL);
	drhoFRdphiLR(0, 0) = dphiFRdphiLR(0, 0) * val(drhoFRdpFR);
	drhoFRdphiLR(1, 0) = dphiFRdphiLR(1, 0) * val(drhoFRdpFR);

	drhoFLdphiLR.col(1).setZero();
	drhoFLdphiLR.col(2).setZero();
	drhoFLdphiLR.col(3).setZero();
	drhoFRdphiLR.col(1).setZero();
	drhoFRdphiLR.col(2).setZero();
	drhoFRdphiLR.col(3).setZero();

	drhoFLdphiLR(0, 4) = dphiFLdphiLR(0, 4) * val(drhoFLdTFL);
	drhoFLdphiLR(1, 4) = dphiFLdphiLR(1, 4) * val(drhoFLdTFL);
	drhoFRdphiLR(0, 4) = dphiFRdphiLR(0, 4) * val(drhoFRdTFR);
	drhoFRdphiLR(1, 4) = dphiFRdphiLR(1, 4) * val(drhoFRdTFR);

	for (int isp = 0; isp < eq_size_spec_; ++isp) {
		drhoFLdphiLR(0, eq_str_spec_ + isp) = dphiFLdphiLR(0, eq_str_spec_ + isp) * val(drhoFLdMFFL[isp]);
		drhoFLdphiLR(1, eq_str_spec_ + isp) = dphiFLdphiLR(1, eq_str_spec_ + isp) * val(drhoFLdMFFL[isp]);
		drhoFRdphiLR(0, eq_str_spec_ + isp) = dphiFRdphiLR(0, eq_str_spec_ + isp) * val(drhoFRdMFFR[isp]);
		drhoFRdphiLR(1, eq_str_spec_ + isp) = dphiFRdphiLR(1, eq_str_spec_ + isp) * val(drhoFRdMFFR[isp]);
	}

	Eigen::MatrixXd dHtFLdphiLR(2, equation_size);
	Eigen::MatrixXd dHtFRdphiLR(2, equation_size);

	dHtFLdphiLR.row(0) <<
		val(dHtFLdpFL * dphiFLdphiLR(0, 0)),
		val(phiFLR(0, 1) * dphiFLdphiLR(0, 1)), val(phiFLR(0, 2) * dphiFLdphiLR(0, 2)), val(phiFLR(0, 3) * dphiFLdphiLR(0, 3)),
		val(dHtFLdTFL * dphiFLdphiLR(0, 4)),
		val(dHtFLdMFFL[0] * dphiFLdphiLR(0, 5));
	dHtFLdphiLR.row(1) <<
		val(dHtFLdpFL * dphiFLdphiLR(1, 0)),
		val(phiFLR(0, 1) * dphiFLdphiLR(1, 1)), val(phiFLR(0, 2) * dphiFLdphiLR(0, 2)), val(phiFLR(0, 3) * dphiFLdphiLR(1, 3)),
		val(dHtFLdTFL * dphiFLdphiLR(1, 4)),
		val(dHtFLdMFFL[0] * dphiFLdphiLR(1, 5));

	dHtFRdphiLR.row(0) <<
		val(dHtFRdpFR * dphiFRdphiLR(0, 0)),
		val(phiFLR(1, 1) * dphiFRdphiLR(0, 1)), val(phiFLR(1, 2) * dphiFRdphiLR(0, 2)), val(phiFLR(1, 3) * dphiFRdphiLR(0, 3)),
		val(dHtFRdTFR * dphiFRdphiLR(0, 4)),
		val(dHtFRdMFFR[0] * dphiFRdphiLR(0, 5));
	dHtFRdphiLR.row(1) <<
		val(dHtFRdpFR * dphiFRdphiLR(1, 0)),
		val(phiFLR(1, 1) * dphiFRdphiLR(1, 1)), val(phiFLR(1, 2) * dphiFRdphiLR(1, 2)), val(phiFLR(1, 3) * dphiFRdphiLR(1, 3)),
		val(dHtFRdTFR * dphiFRdphiLR(1, 4)),
		val(dHtFRdMFFR[0] * dphiFRdphiLR(1, 5));


	auto calc_dcdp = [](
		const auto& c,
		const auto& rho,
		const auto& drhodp,
		const auto& drhodT,
		const auto& drhodpdp,
		const auto& drhodpdT,
		const auto& drhodTdT,
		const auto& dHtdp,
		const auto& dHtdT
		) {
			double dAdp =
				(drhodpdp + drhodpdT * (1.0 / rho - dHtdp) / dHtdT +
					drhodT * (-1.0 / rho / rho * drhodp) / dHtdT);
			return -0.5 * c * c * c * dAdp;

		};

	double dcFLdpFLR_coeff = calc_dcdp(
		val(cFL), val(rhoFL), val(drhoFLdpFL), val(drhoFLdTFL),
		val(drhoFLdpdpFL), val(drhoFLdpdTFL), val(drhoFLdTdTFL), val(dHtFLdpFL), val(dHtFLdTFL));
	double dcFLdpL = dphiFLdphiLR(0, 0) * dcFLdpFLR_coeff;
	double dcFLdpR = dphiFLdphiLR(1, 0) * dcFLdpFLR_coeff;

	double dcFRdpFLR_coeff = calc_dcdp(
		val(cFR), val(rhoFR), val(drhoFRdpFR), val(drhoFRdTFR),
		val(drhoFRdpdpFR), val(drhoFRdpdTFR), val(drhoFRdTdTFR), val(dHtFRdpFR), val(dHtFRdTFR));
	double dcFRdpL = dphiFRdphiLR(0, 0) * dcFRdpFLR_coeff;
	double dcFRdpR = dphiFRdphiLR(1, 0) * dcFRdpFLR_coeff;



	auto calc_dcdT = [](
		const auto& c,
		const auto& rho,
		const auto& drhodp,
		const auto& drhodT,
		const auto& drhodpdp,
		const auto& drhodpdT,
		const auto& drhodTdT,
		const auto& dHtdp,
		const auto& dHtdT
		) {
			double dAdT =
				(drhodpdT + drhodTdT * (1.0 / rho - dHtdp) / dHtdT +
					drhodT * (-1.0 / rho / rho * drhodT) / dHtdT);
			return -0.5 * c * c * c * dAdT;
		};
	double dcFLdTFLR_coeff = calc_dcdT(
		val(cFL), val(rhoFL), val(drhoFLdpFL), val(drhoFLdTFL),
		val(drhoFLdpdpFL), val(drhoFLdpdTFL), val(drhoFLdTdTFL), val(dHtFLdpFL), val(dHtFLdTFL));
	double dcFLdTL = dphiFLdphiLR(0, 4) * dcFLdTFLR_coeff;
	double dcFLdTR = dphiFLdphiLR(1, 4) * dcFLdTFLR_coeff;

	double dcFRdTFLR_coeff = calc_dcdT(
		val(cFR), val(rhoFR), val(drhoFRdpFR), val(drhoFRdTFR),
		val(drhoFRdpdpFR), val(drhoFRdpdTFR), val(drhoFRdTdTFR), val(dHtFRdpFR), val(dHtFRdTFR));
	double dcFRdTL = dphiFRdphiLR(0, 4) * dcFRdTFLR_coeff;
	double dcFRdTR = dphiFRdphiLR(1, 4) * dcFRdTFLR_coeff;



	auto calc_dcdMF = [](
		const auto& c,
		const auto& rho,
		const auto& drhodp,
		const auto& drhodT,
		const auto& drhodpdp,
		const auto& drhodpdT,
		const auto& drhodTdT,
		const auto& dHtdp,
		const auto& dHtdT,
		const auto& drhodpdMF,
		const auto& drhodTdMF,
		const auto& drhodMF,
		const auto& dHtdpdMF,
		const auto& dHtdTdMF
		) {
			double dAdT =
				(drhodpdMF + drhodTdMF * (1.0 / rho - dHtdp) / dHtdT +
					drhodT * (-1.0 / rho / rho * drhodMF - dHtdpdMF) / dHtdT -
					drhodT * (1.0 / rho - dHtdp) / dHtdT / dHtdT * dHtdTdMF);
			return -0.5 * c * c * c * dAdT;
		};

	std::vector<double> dcFLdMFL(eq_size_spec_), dcFLdMFR(eq_size_spec_);
	std::vector<double> dcFRdMFL(eq_size_spec_), dcFRdMFR(eq_size_spec_);
	for (int isp = 0; isp < eq_size_spec_; ++isp) {
		double dcFLdMFFLR_coeff = calc_dcdMF(
			val(cFL), val(rhoFL), val(drhoFLdpFL), val(drhoFLdTFL),
			val(drhoFLdpdpFL), val(drhoFLdpdTFL), val(drhoFLdTdTFL), val(dHtFLdpFL), val(dHtFLdTFL),
			val(drhoFLdpdMFFL[isp]), val(drhoFLdTdMFFL[isp]), val(drhoFLdMFFL[isp]), val(dHtdpdMFFL[isp]), val(dHtdTdMFFL[isp]));
		dcFLdMFL[isp] = dphiFLdphiLR(0, eq_str_spec_ + isp) * dcFLdMFFLR_coeff;
		dcFLdMFR[isp] = dphiFLdphiLR(1, eq_str_spec_ + isp) * dcFLdMFFLR_coeff;

		double dcFRdMFFLR_coeff = calc_dcdMF(
			val(cFR), val(rhoFR), val(drhoFRdpFR), val(drhoFRdTFR),
			val(drhoFRdpdpFR), val(drhoFRdpdTFR), val(drhoFRdTdTFR), val(dHtFRdpFR), val(dHtFRdTFR),
			val(drhoFRdpdMFFR[isp]), val(drhoFRdTdMFFR[isp]), val(drhoFRdMFFR[isp]), val(dHtdpdMFFR[isp]), val(dHtdTdMFFR[isp]));
		dcFRdMFL[isp] = dphiFRdphiLR(0, eq_str_spec_ + isp) * dcFRdMFFLR_coeff;
		dcFRdMFR[isp] = dphiFRdphiLR(1, eq_str_spec_ + isp) * dcFRdMFFLR_coeff;

	}

	//=======================================================




	//std::cout << autodiff::gradient(rhoFR, W) << std::endl << std::endl;

	//double drhoFRdpL = val(drhodpFR) * DWF_pR;
	//double drhoFRdTL = val(drhodTFR) * DWF_TR;
	//double drhoFRdY0L = val(drhodMFFR[0]) * DWF_Y0R;
	//double drhoFRdpR = val(drhodpFR) * (1.0 - DWF_pR);
	//double drhoFRdTR = val(drhodTFR) * (1.0 - DWF_TR);
	//double drhoFRdY0R = val(drhodMFFR[0]) * (1.0 - DWF_Y0R);
	//std::cout << drhoFRdpL << std::endl << std::endl;
	//std::cout << drhoFRdTL << std::endl << std::endl;
	//std::cout << drhoFRdY0L << std::endl << std::endl;
	//std::cout << drhoFRdpR << std::endl << std::endl;
	//std::cout << drhoFRdTR << std::endl << std::endl;
	//std::cout << drhoFRdY0R << std::endl << std::endl;

	//std::cout << autodiff::gradient(cFL, W) << std::endl << std::endl;

	//{
	//	autodiff::var dAdp =
	//		(drhodpdpFL + drhodpdTFL * (1.0 / rhoFL - dHtdpFL) / dHtdTFL +
	//			drhodTFL * (-1.0 / rhoFL / rhoFL * drhodpFL) / dHtdTFL);
	//	double cFLdpFL = -0.5 * val(cFL * cFL * cFL * dAdp) * (1.0 - DWF_pL);
	//	std::cout << cFLdpFL << std::endl << std::endl;
	//}
	//{
	//	autodiff::var dAdT =
	//		(drhodpdTFL + drhodTdTFL * (1.0 / rhoFL - dHtdpFL) / dHtdTFL +
	//			drhodTFL * (-1.0 / rhoFL / rhoFL * drhodTFL) / dHtdTFL);
	//	double cFLdTFL = -0.5 * val(cFL * cFL * cFL * dAdT) * (1.0 - DWF_TL);
	//	std::cout << cFLdTFL << std::endl << std::endl;
	//}
	//{
	//	autodiff::var dAdY0 =
	//		(drhoFLdpdMFFL[0] + drhoFLdTdMFFL[0] * (1.0 / rhoFL - dHtdpFL) / dHtdTFL +
	//			drhodTFL * (-1.0 / rhoFL / rhoFL * drhodMFFL[0] - dHtdpdMFFL[0]) / dHtdTFL -
	//			drhodTFL * (1.0 / rhoFL - dHtdpFL) / dHtdTFL / dHtdTFL * dHtdTdMFFL[0]);
	//	double cFLdY0FL = -0.5 * val(cFL * cFL * cFL * dAdY0) * (1.0 - DWF_Y0L);
	//	std::cout << cFLdY0FL << std::endl << std::endl;
	//}
	//{
	//	autodiff::var dAdp =
	//		(drhodpdpFL + drhodpdTFL * (1.0 / rhoFL - dHtdpFL) / dHtdTFL +
	//			drhodTFL * (-1.0 / rhoFL / rhoFL * drhodpFL) / dHtdTFL);
	//	double cFLdpFL = -0.5 * val(cFL * cFL * cFL * dAdp) * DWF_pL;
	//	std::cout << cFLdpFL << std::endl << std::endl;
	//}
	//{
	//	autodiff::var dAdT =
	//		(drhodpdTFL + drhodTdTFL * (1.0 / rhoFL - dHtdpFL) / dHtdTFL +
	//			drhodTFL * (-1.0 / rhoFL / rhoFL * drhodTFL) / dHtdTFL);
	//	double cFLdTFL = -0.5 * val(cFL * cFL * cFL * dAdT) * DWF_TL;
	//	std::cout << cFLdTFL << std::endl << std::endl;
	//}


	// AUSM-like expression of HLLC and its all-speed extension
	auto UnFL = phiFLR(0, 1) * nvec[0] + phiFLR(0, 2) * nvec[1] + phiFLR(0, 3) * nvec[2];
	auto UnFR = phiFLR(1, 1) * nvec[0] + phiFLR(1, 2) * nvec[1] + phiFLR(1, 3) * nvec[2];
	auto chat = 0.5 * (cFL + cFR);
	auto rhohat = 0.5 * (rhoFL + rhoFR);

	autodiff::var ML = UnFL / chat;
	autodiff::var MR = UnFR / chat;

	auto U2L = phiFLR(0, 1) * phiFLR(0, 1) + phiFLR(0, 2) * phiFLR(0, 2) + phiFLR(0, 3) * phiFLR(0, 3);
	auto U2R = phiFLR(1, 1) * phiFLR(1, 1) + phiFLR(1, 2) * phiFLR(1, 2) + phiFLR(1, 3) * phiFLR(1, 3);
	autodiff::var KLR = sqrt(0.5 * (U2L + U2R));
	auto Mk = min(1.0, KLR / chat);
	autodiff::var Xi = (1.0 - Mk) * (1.0 - Mk);

	auto MLP = 0.5 * (ML + abs(ML));
	if (abs(ML) < 1.0) MLP = 0.25 * (ML + 1.0) * (ML + 1.0);
	auto MRM = 0.5 * (MR - abs(MR));
	if (abs(MR) < 1.0) MRM = -0.25 * (MR - 1.0) * (MR - 1.0);
	autodiff::var PLP = 0.5 * (1.0 + (ML > 0.0 ? 1.0 : -1.0));
	if (abs(ML) < 1.0) PLP = 0.25 * (ML + 1.0) * (ML + 1.0) * (2.0 - ML);
	autodiff::var PRM = 0.5 * (1.0 - (MR > 0.0 ? 1.0 : -1.0));
	if (abs(MR) < 1.0) PRM = 0.25 * (MR - 1.0) * (MR - 1.0) * (2.0 + MR);
	auto Ubar = (rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR);
	autodiff::var g_c = 1.0 + max(min(ML, 0.0), -1.0) * min(max(MR, 0.0), 1.0);
	auto D_L = UnFL + (1.0 - g_c) * abs(UnFL);
	auto D_R = UnFR - (1.0 - g_c) * abs(UnFR);
	auto D_rho = Ubar * g_c;
	autodiff::var UPL = D_L + D_rho;
	autodiff::var UMR = D_R - D_rho;

	// HAUS mass flux
	autodiff::var mdot = 0.5 * rhoFL * UPL + 0.5 * rhoFR * UMR - 0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat;

	//// HAUS pressure flux
	//autodiff::var pF = 0.5 * (phiFLR(0, 0) + phiFLR(1, 0)) - 0.5 * (PLP - PRM) * (phiFLR(1, 0) - phiFLR(0, 0)) +
	//	KLR * (PLP + PRM - 1.0) * rhohat * chat;
	double Mlim = 0.01;
	auto Mo = min(1.0, max(Mk, Mlim));
	autodiff::var fa = Mo * (2.0 - Mo);
	double Ku = 0.75;
	//pF = pF - Ku * PLP * PRM * fa * rhohat * chat * (UnFR - UnFL);

	//autodiff::var flux0 = mdot;
	//autodiff::var flux1 = max(0, mdot) * phiFLR(0, 1) + min(0, mdot) * phiFLR(1, 1) + nvec[0] * pF;
	//autodiff::var flux2 = max(0, mdot) * phiFLR(0, 2) + min(0, mdot) * phiFLR(1, 2) + nvec[1] * pF;
	//autodiff::var flux3 = max(0, mdot) * phiFLR(0, 3) + min(0, mdot) * phiFLR(1, 3) + nvec[2] * pF;
	//autodiff::var flux4 = max(0, mdot) * HtFL + min(0, mdot) * HtFR;
	//autodiff::var flux5 = max(0, mdot) * phiFLR(0, 5) + min(0, mdot) * phiFLR(1, 5);

	//auto dFdW = autodiff::gradient(pF, W);

	//std::cout << dFdW << std::endl << std::endl;


	//--------------------------------------

	double dUnFLduL = dphiFLdphiLR(0, 1) * nvec[0]; double dUnFLduR = dphiFLdphiLR(1, 1) * nvec[0];
	double dUnFRduL = dphiFRdphiLR(0, 1) * nvec[0]; double dUnFRduR = dphiFRdphiLR(1, 1) * nvec[0];
	double dUnFLdvL = dphiFLdphiLR(0, 2) * nvec[1]; double dUnFLdvR = dphiFLdphiLR(1, 2) * nvec[1];
	double dUnFRdvL = dphiFRdphiLR(0, 2) * nvec[1]; double dUnFRdvR = dphiFRdphiLR(1, 2) * nvec[1];
	double dUnFLdwL = dphiFLdphiLR(0, 3) * nvec[2]; double dUnFLdwR = dphiFLdphiLR(1, 3) * nvec[2];
	double dUnFRdwL = dphiFRdphiLR(0, 3) * nvec[2]; double dUnFRdwR = dphiFRdphiLR(1, 3) * nvec[2];

	Eigen::MatrixXd dMLdphiLR(2, equation_size);
	Eigen::MatrixXd dMRdphiLR(2, equation_size);

	dMLdphiLR(0, 0) = -val(UnFL / chat / chat) * 0.5 * (dcFLdpL + dcFRdpL); dMLdphiLR(1, 0) = -val(UnFL / chat / chat) * 0.5 * (dcFLdpR + dcFRdpR);
	dMLdphiLR(0, 1) = val(dUnFLduL / chat); dMLdphiLR(1, 1) = val(dUnFLduR / chat);
	dMLdphiLR(0, 2) = val(dUnFLdvL / chat); dMLdphiLR(1, 2) = val(dUnFLdvR / chat);
	dMLdphiLR(0, 3) = val(dUnFLdwL / chat); dMLdphiLR(1, 3) = val(dUnFLdwR / chat);
	dMLdphiLR(0, 4) = -val(UnFL / chat / chat) * 0.5 * (dcFLdTL + dcFRdTL); dMLdphiLR(1, 4) = -val(UnFL / chat / chat) * 0.5 * (dcFLdTR + dcFRdTR);
	for (int isp = 0; isp < eq_size_spec_; ++isp) {
		dMLdphiLR(0, eq_str_spec_ + isp) = -val(UnFL / chat / chat) * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]);
		dMLdphiLR(1, eq_str_spec_ + isp) = -val(UnFL / chat / chat) * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]);
	}

	dMRdphiLR(0, 0) = -val(UnFR / chat / chat) * 0.5 * (dcFLdpL + dcFRdpL); dMRdphiLR(1, 0) = -val(UnFR / chat / chat) * 0.5 * (dcFLdpR + dcFRdpR);
	dMRdphiLR(0, 1) = val(dUnFRduL / chat); dMRdphiLR(1, 1) = val(dUnFRduR / chat);
	dMRdphiLR(0, 2) = val(dUnFRdvL / chat); dMRdphiLR(1, 2) = val(dUnFRdvR / chat);
	dMRdphiLR(0, 3) = val(dUnFRdwL / chat); dMRdphiLR(1, 3) = val(dUnFRdwR / chat);
	dMRdphiLR(0, 4) = -val(UnFR / chat / chat) * 0.5 * (dcFLdTL + dcFRdTL); dMRdphiLR(1, 4) = -val(UnFR / chat / chat) * 0.5 * (dcFLdTR + dcFRdTR);
	for (int isp = 0; isp < eq_size_spec_; ++isp) {
		dMRdphiLR(0, eq_str_spec_ + isp) = -val(UnFR / chat / chat) * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]);
		dMRdphiLR(1, eq_str_spec_ + isp) = -val(UnFR / chat / chat) * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]);
	}


	//auto soft_clip = [](auto x, auto a, auto b, auto eps) {
	//	const double xm = 0.5 * (a + b);
	//	const double dx = 0.5 * (b - a);
	//	return xm + dx * tanh((x - xm) * 2.0 / eps);
	//};


	Eigen::MatrixXd dg_cdphiLR(2, equation_size);
	dg_cdphiLR.setZero();
	if (ML > -1.0 && ML < 0.0) dg_cdphiLR += dMLdphiLR * val(min(max(MR, 0.0), 1.0));
	if (MR > 0.0 && MR < 1.0) dg_cdphiLR += dMRdphiLR * val(max(min(ML, 0.0), -1.0));


	Eigen::MatrixXd dUPLdphiLR(2, equation_size);
	Eigen::MatrixXd dUMRdphiLR(2, equation_size);

	double dUbardpL =
		val((drhoFLdphiLR(0, 0) * abs(UnFL) + drhoFRdphiLR(0, 0) * abs(UnFR)) / (rhoFL + rhoFR) -
			(rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR) / (rhoFL + rhoFR) *
			(drhoFLdphiLR(0, 0) + drhoFRdphiLR(0, 0)));
	double dD_LdpL = val(-dg_cdphiLR(0, 0) * abs(UnFL));
	double dD_rhodpL = val(dUbardpL * g_c + Ubar * dg_cdphiLR(0, 0));
	double dD_RdpL = val(dg_cdphiLR(0, 0) * abs(UnFR));
	dUPLdphiLR(0, 0) = dD_LdpL + dD_rhodpL;
	dUMRdphiLR(0, 0) = dD_RdpL - dD_rhodpL;

	double dUbardpR =
		val((drhoFLdphiLR(1, 0) * abs(UnFL) + drhoFRdphiLR(1, 0) * abs(UnFR)) / (rhoFL + rhoFR) -
			(rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR) / (rhoFL + rhoFR) *
			(drhoFLdphiLR(1, 0) + drhoFRdphiLR(1, 0)));
	double dD_LdpR = val(-dg_cdphiLR(1, 0) * abs(UnFL));
	double dD_rhodpR = val(dUbardpR * g_c + Ubar * dg_cdphiLR(1, 0));
	double dD_RdpR = val(dg_cdphiLR(1, 0) * abs(UnFR));
	dUPLdphiLR(1, 0) = dD_LdpR + dD_rhodpR;
	dUMRdphiLR(1, 0) = dD_RdpR - dD_rhodpR;

	double dUbardTL =
		val((drhoFLdphiLR(0, 4) * abs(UnFL) + drhoFRdphiLR(0, 4) * abs(UnFR)) / (rhoFL + rhoFR) -
			(rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR) / (rhoFL + rhoFR) *
			(drhoFLdphiLR(0, 4) + drhoFRdphiLR(0, 4)));
	double dD_LdTL = val(-dg_cdphiLR(0, 4) * abs(UnFL));
	double dD_rhodTL = val(dUbardTL * g_c + Ubar * dg_cdphiLR(0, 4));
	double dD_RdTL = val(dg_cdphiLR(0, 4) * abs(UnFR));
	dUPLdphiLR(0, 4) = dD_LdTL + dD_rhodTL;
	dUMRdphiLR(0, 4) = dD_RdTL - dD_rhodTL;

	double dUbardTR =
		val((drhoFLdphiLR(1, 4) * abs(UnFL) + drhoFRdphiLR(1, 4) * abs(UnFR)) / (rhoFL + rhoFR) -
			(rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR) / (rhoFL + rhoFR) *
			(drhoFLdphiLR(1, 4) + drhoFRdphiLR(1, 4)));
	double dD_LdTR = val(-dg_cdphiLR(1, 4) * abs(UnFL));
	double dD_rhodTR = val(dUbardTR * g_c + Ubar * dg_cdphiLR(1, 4));
	double dD_RdTR = val(dg_cdphiLR(1, 4) * abs(UnFR));
	dUPLdphiLR(1, 4) = dD_LdTR + dD_rhodTR;
	dUMRdphiLR(1, 4) = dD_RdTR - dD_rhodTR;


	double dabsUnFLduL = dUnFLduL * (UnFL > 0.0 ? 1.0 : -1.0);
	double dabsUnFRduL = dUnFRduL * (UnFR > 0.0 ? 1.0 : -1.0);
	double dD_LduL = val(dUnFLduL - dg_cdphiLR(0, 1) * abs(UnFL) + (1.0 - g_c) * dabsUnFLduL);
	double dD_RduL = val(dUnFRduL + dg_cdphiLR(0, 1) * abs(UnFR) - (1.0 - g_c) * dabsUnFRduL);
	double dD_rhoduL = val(dg_cdphiLR(0, 1) * Ubar + g_c * (rhoFL * dabsUnFLduL + rhoFR * dabsUnFRduL) / (rhoFL + rhoFR));
	dUPLdphiLR(0, 1) = dD_LduL + dD_rhoduL;
	dUMRdphiLR(0, 1) = dD_RduL - dD_rhoduL;

	double dabsUnFLduR = dUnFLduR * (UnFL > 0.0 ? 1.0 : -1.0);
	double dabsUnFRduR = dUnFRduR * (UnFR > 0.0 ? 1.0 : -1.0);
	double dD_LduR = val(dUnFLduR - dg_cdphiLR(1, 1) * abs(UnFL) + (1.0 - g_c) * dabsUnFLduR);
	double dD_RduR = val(dUnFRduR + dg_cdphiLR(1, 1) * abs(UnFR) - (1.0 - g_c) * dabsUnFRduR);
	double dD_rhoduR = val(dg_cdphiLR(1, 1) * Ubar + g_c * (rhoFL * dabsUnFLduR + rhoFR * dabsUnFRduR) / (rhoFL + rhoFR));
	dUPLdphiLR(1, 1) = dD_LduR + dD_rhoduR;
	dUMRdphiLR(1, 1) = dD_RduR - dD_rhoduR;

	double dabsUnFLdvL = dUnFLdvL * (UnFL > 0.0 ? 1.0 : -1.0);
	double dabsUnFRdvL = dUnFRdvL * (UnFR > 0.0 ? 1.0 : -1.0);
	double dD_LdvL = val(dUnFLdvL - dg_cdphiLR(0, 2) * abs(UnFL) + (1.0 - g_c) * dabsUnFLdvL);
	double dD_RdvL = val(dUnFRdvL + dg_cdphiLR(0, 2) * abs(UnFR) - (1.0 - g_c) * dabsUnFRdvL);
	double dD_rhodvL = val(dg_cdphiLR(0, 2) * Ubar + g_c * (rhoFL * dabsUnFLdvL + rhoFR * dabsUnFRdvL) / (rhoFL + rhoFR));
	dUPLdphiLR(0, 2) = dD_LdvL + dD_rhodvL;
	dUMRdphiLR(0, 2) = dD_RdvL - dD_rhodvL;

	double dabsUnFLdvR = dUnFLdvR * (UnFL > 0.0 ? 1.0 : -1.0);
	double dabsUnFRdvR = dUnFRdvR * (UnFR > 0.0 ? 1.0 : -1.0);
	double dD_LdvR = val(dUnFLdvR - dg_cdphiLR(1, 2) * abs(UnFL) + (1.0 - g_c) * dabsUnFLdvR);
	double dD_RdvR = val(dUnFRdvR + dg_cdphiLR(1, 2) * abs(UnFR) - (1.0 - g_c) * dabsUnFRdvR);
	double dD_rhodvR = val(dg_cdphiLR(1, 2) * Ubar + g_c * (rhoFL * dabsUnFLdvR + rhoFR * dabsUnFRdvR) / (rhoFL + rhoFR));
	dUPLdphiLR(1, 2) = dD_LdvR + dD_rhodvR;
	dUMRdphiLR(1, 2) = dD_RdvR - dD_rhodvR;

	double dabsUnFLdwL = dUnFLdwL * (UnFL > 0.0 ? 1.0 : -1.0);
	double dabsUnFRdwL = dUnFRdwL * (UnFR > 0.0 ? 1.0 : -1.0);
	double dD_LdwL = val(dUnFLdwL - dg_cdphiLR(0, 3) * abs(UnFL) + (1.0 - g_c) * dabsUnFLdwL);
	double dD_RdwL = val(dUnFRdwL + dg_cdphiLR(0, 3) * abs(UnFR) - (1.0 - g_c) * dabsUnFRdwL);
	double dD_rhodwL = val(dg_cdphiLR(0, 3) * Ubar + g_c * (rhoFL * dabsUnFLdwL + rhoFR * dabsUnFRdwL) / (rhoFL + rhoFR));
	dUPLdphiLR(0, 3) = dD_LdwL + dD_rhodwL;
	dUMRdphiLR(0, 3) = dD_RdwL - dD_rhodwL;

	double dabsUnFLdwR = dUnFLdwR * (UnFL > 0.0 ? 1.0 : -1.0);
	double dabsUnFRdwR = dUnFRdwR * (UnFR > 0.0 ? 1.0 : -1.0);
	double dD_LdwR = val(dUnFLdwR - dg_cdphiLR(1, 3) * abs(UnFL) + (1.0 - g_c) * dabsUnFLdwR);
	double dD_RdwR = val(dUnFRdwR + dg_cdphiLR(1, 3) * abs(UnFR) - (1.0 - g_c) * dabsUnFRdwR);
	double dD_rhodwR = val(dg_cdphiLR(1, 3) * Ubar + g_c * (rhoFL * dabsUnFLdwR + rhoFR * dabsUnFRdwR) / (rhoFL + rhoFR));
	dUPLdphiLR(1, 3) = dD_LdwR + dD_rhodwR;
	dUMRdphiLR(1, 3) = dD_RdwR - dD_rhodwR;


	for (int isp = 0; isp < eq_size_spec_; ++isp) {
		double dUbardMFL =
			val((drhoFLdphiLR(0, eq_str_spec_ + isp) * abs(UnFL) + drhoFRdphiLR(0, eq_str_spec_ + isp) * abs(UnFR)) / (rhoFL + rhoFR) -
				(rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR) / (rhoFL + rhoFR) *
				(drhoFLdphiLR(0, eq_str_spec_ + isp) + drhoFRdphiLR(0, eq_str_spec_ + isp)));
		double dD_LdMFL = val(-dg_cdphiLR(0, eq_str_spec_ + isp) * abs(UnFL));
		double dD_rhodMFL = val(dUbardMFL * g_c + Ubar * dg_cdphiLR(0, eq_str_spec_ + isp));
		double dD_RdMFL = val(dg_cdphiLR(0, eq_str_spec_ + isp) * abs(UnFR));
		dUPLdphiLR(0, eq_str_spec_ + isp) = dD_LdMFL + dD_rhodMFL;
		dUMRdphiLR(0, eq_str_spec_ + isp) = dD_RdMFL - dD_rhodMFL;

		double dUbardMFR =
			val((drhoFLdphiLR(1, eq_str_spec_ + isp) * abs(UnFL) + drhoFRdphiLR(1, eq_str_spec_ + isp) * abs(UnFR)) / (rhoFL + rhoFR) -
				(rhoFL * abs(UnFL) + rhoFR * abs(UnFR)) / (rhoFL + rhoFR) / (rhoFL + rhoFR) *
				(drhoFLdphiLR(1, eq_str_spec_ + isp) + drhoFRdphiLR(1, eq_str_spec_ + isp)));
		double dD_LdMFR = val(-dg_cdphiLR(1, eq_str_spec_ + isp) * abs(UnFL));
		double dD_rhodMFR = val(dUbardMFR * g_c + Ubar * dg_cdphiLR(1, eq_str_spec_ + isp));
		double dD_RdMFR = val(dg_cdphiLR(1, eq_str_spec_ + isp) * abs(UnFR));
		dUPLdphiLR(1, eq_str_spec_ + isp) = dD_LdMFR + dD_rhodMFR;
		dUMRdphiLR(1, eq_str_spec_ + isp) = dD_RdMFR - dD_rhodMFR;
	}

	double dKLRduL = val(0.5 / KLR * 0.5 * (2.0 * phiFLR(0, 1) * dphiFLdphiLR(0, 1) + 2.0 * phiFLR(1, 1) * dphiFRdphiLR(0, 1)));
	double dKLRdvL = val(0.5 / KLR * 0.5 * (2.0 * phiFLR(0, 2) * dphiFLdphiLR(0, 2) + 2.0 * phiFLR(1, 2) * dphiFRdphiLR(0, 2)));
	double dKLRdwL = val(0.5 / KLR * 0.5 * (2.0 * phiFLR(0, 3) * dphiFLdphiLR(0, 3) + 2.0 * phiFLR(1, 3) * dphiFRdphiLR(0, 3)));

	double dKLRduR = val(0.5 / KLR * 0.5 * (2.0 * phiFLR(0, 1) * dphiFLdphiLR(1, 1) + 2.0 * phiFLR(1, 1) * dphiFRdphiLR(1, 1)));
	double dKLRdvR = val(0.5 / KLR * 0.5 * (2.0 * phiFLR(0, 2) * dphiFLdphiLR(1, 2) + 2.0 * phiFLR(1, 2) * dphiFRdphiLR(1, 2)));
	double dKLRdwR = val(0.5 / KLR * 0.5 * (2.0 * phiFLR(0, 3) * dphiFLdphiLR(1, 3) + 2.0 * phiFLR(1, 3) * dphiFRdphiLR(1, 3)));

	Eigen::MatrixXd dXidphiLR(2, equation_size);
	dXidphiLR.setZero();
	if (KLR / chat < 1.0) {
		dXidphiLR(0, 0) = val(2.0 * (1.0 - Mk) * KLR / chat / chat * 0.5 * (dcFLdpL + dcFRdpL));
		dXidphiLR(0, 1) = val(-2.0 * (1.0 - Mk) / chat * dKLRduL);
		dXidphiLR(0, 2) = val(-2.0 * (1.0 - Mk) / chat * dKLRdvL);
		dXidphiLR(0, 3) = val(-2.0 * (1.0 - Mk) / chat * dKLRdwL);
		dXidphiLR(0, 4) = val(2.0 * (1.0 - Mk) * KLR / chat / chat * 0.5 * (dcFLdTL + dcFRdTL));
		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dXidphiLR(0, eq_str_spec_ + isp) = val(2.0 * (1.0 - Mk) * KLR / chat / chat * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]));
		}

		dXidphiLR(1, 0) = val(2.0 * (1.0 - Mk) * KLR / chat / chat * 0.5 * (dcFLdpR + dcFRdpR));
		dXidphiLR(1, 1) = val(-2.0 * (1.0 - Mk) / chat * dKLRduR);
		dXidphiLR(1, 2) = val(-2.0 * (1.0 - Mk) / chat * dKLRdvR);
		dXidphiLR(1, 3) = val(-2.0 * (1.0 - Mk) / chat * dKLRdwR);
		dXidphiLR(1, 4) = val(2.0 * (1.0 - Mk) * KLR / chat / chat * 0.5 * (dcFLdTR + dcFRdTR));
		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dXidphiLR(1, eq_str_spec_ + isp) = val(2.0 * (1.0 - Mk) * KLR / chat / chat * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]));
		}
	}

	Eigen::VectorXd dmdotdphiL(equation_size);
	Eigen::VectorXd dmdotdphiR(equation_size);

	{
		dmdotdphiL[0] =
			val(
				0.5 * drhoFLdphiLR(0, 0) * UPL + 0.5 * drhoFRdphiLR(0, 0) * UMR +
				0.5 * rhoFL * dUPLdphiLR(0, 0) + 0.5 * rhoFR * dUMRdphiLR(0, 0) -
				0.5 * dXidphiLR(0, 0) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat -
				0.5 * Xi * (dphiFRdphiLR(0, 0) - dphiFLdphiLR(0, 0)) / chat +
				0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat / chat * 0.5 * (dcFLdpL + dcFRdpL));

		dmdotdphiL[1] = val(0.5 * rhoFL * dUPLdphiLR(0, 1) + 0.5 * rhoFR * dUMRdphiLR(0, 1) - 0.5 * dXidphiLR(0, 1) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat);
		dmdotdphiL[2] = val(0.5 * rhoFL * dUPLdphiLR(0, 2) + 0.5 * rhoFR * dUMRdphiLR(0, 2) - 0.5 * dXidphiLR(0, 2) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat);
		dmdotdphiL[3] = val(0.5 * rhoFL * dUPLdphiLR(0, 3) + 0.5 * rhoFR * dUMRdphiLR(0, 3) - 0.5 * dXidphiLR(0, 3) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat);

		dmdotdphiL[4] =
			val(
				0.5 * drhoFLdphiLR(0, 4) * UPL + 0.5 * drhoFRdphiLR(0, 4) * UMR +
				0.5 * rhoFL * dUPLdphiLR(0, 4) + 0.5 * rhoFR * dUMRdphiLR(0, 4) -
				0.5 * dXidphiLR(0, 4) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat +
				0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat / chat * 0.5 * (dcFLdTL + dcFRdTL));

		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dmdotdphiL[eq_str_spec_ + isp] =
				val(
					0.5 * drhoFLdphiLR(0, eq_str_spec_ + isp) * UPL + 0.5 * drhoFRdphiLR(0, eq_str_spec_ + isp) * UMR +
					0.5 * rhoFL * dUPLdphiLR(0, eq_str_spec_ + isp) + 0.5 * rhoFR * dUMRdphiLR(0, eq_str_spec_ + isp) -
					0.5 * dXidphiLR(0, eq_str_spec_ + isp) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat +
					0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat / chat * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]));
		}
	}



	{
		dmdotdphiR[0] =
			val(
				0.5 * drhoFLdphiLR(1, 0) * UPL + 0.5 * drhoFRdphiLR(1, 0) * UMR +
				0.5 * rhoFL * dUPLdphiLR(1, 0) + 0.5 * rhoFR * dUMRdphiLR(1, 0) -
				0.5 * dXidphiLR(1, 0) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat -
				0.5 * Xi * (dphiFRdphiLR(1, 0) - dphiFLdphiLR(1, 0)) / chat +
				0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat / chat * 0.5 * (dcFLdpR + dcFRdpR));

		dmdotdphiR[1] = val(0.5 * rhoFL * dUPLdphiLR(1, 1) + 0.5 * rhoFR * dUMRdphiLR(1, 1) - 0.5 * dXidphiLR(1, 1) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat);
		dmdotdphiR[2] = val(0.5 * rhoFL * dUPLdphiLR(1, 2) + 0.5 * rhoFR * dUMRdphiLR(1, 2) - 0.5 * dXidphiLR(1, 2) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat);
		dmdotdphiR[3] = val(0.5 * rhoFL * dUPLdphiLR(1, 3) + 0.5 * rhoFR * dUMRdphiLR(1, 3) - 0.5 * dXidphiLR(1, 3) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat);

		dmdotdphiR[4] =
			val(
				0.5 * drhoFLdphiLR(1, 4) * UPL + 0.5 * drhoFRdphiLR(1, 4) * UMR +
				0.5 * rhoFL * dUPLdphiLR(1, 4) + 0.5 * rhoFR * dUMRdphiLR(1, 4) -
				0.5 * dXidphiLR(1, 4) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat +
				0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat / chat * 0.5 * (dcFLdTR + dcFRdTR));

		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dmdotdphiR[eq_str_spec_ + isp] =
				val(
					0.5 * drhoFLdphiLR(1, eq_str_spec_ + isp) * UPL + 0.5 * drhoFRdphiLR(1, eq_str_spec_ + isp) * UMR +
					0.5 * rhoFL * dUPLdphiLR(1, eq_str_spec_ + isp) + 0.5 * rhoFR * dUMRdphiLR(1, eq_str_spec_ + isp) -
					0.5 * dXidphiLR(1, eq_str_spec_ + isp) * (phiFLR(1, 0) - phiFLR(0, 0)) / chat +
					0.5 * Xi * (phiFLR(1, 0) - phiFLR(0, 0)) / chat / chat * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]));
		}
	}



	//------------------------------

	Eigen::MatrixXd dModphiLR(2, equation_size);
	dModphiLR.setZero();

	if (Mk > Mlim && Mk < 1.0) {
		dModphiLR(0, 0) = val(-KLR / chat / chat * 0.5 * (dcFLdpL + dcFRdpL));
		dModphiLR(0, 1) = val(dKLRduL / chat);
		dModphiLR(0, 2) = val(dKLRdvL / chat);
		dModphiLR(0, 3) = val(dKLRdwL / chat);
		dModphiLR(0, 4) = val(-KLR / chat / chat * 0.5 * (dcFLdTL + dcFRdTL));
		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dModphiLR(0, eq_str_spec_ + isp) = val(-KLR / chat / chat * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]));
		}

		dModphiLR(1, 0) = val(-KLR / chat / chat * 0.5 * (dcFLdpR + dcFRdpR));
		dModphiLR(1, 1) = val(dKLRduR / chat);
		dModphiLR(1, 2) = val(dKLRdvR / chat);
		dModphiLR(1, 3) = val(dKLRdwR / chat);
		dModphiLR(1, 4) = val(-KLR / chat / chat * 0.5 * (dcFLdTR + dcFRdTR));
		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dModphiLR(1, eq_str_spec_ + isp) = val(-KLR / chat / chat * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]));
		}
	}

	Eigen::MatrixXd dfadphiLR = val((2.0 - 2.0 * Mo)) * dModphiLR;

	Eigen::MatrixXd dPLPdphiLR(2, equation_size);
	dPLPdphiLR.setZero();
	if (abs(ML) < 1.0) {
		dPLPdphiLR = val(Ku * (ML + 1.0) * (1.0 - ML)) * dMLdphiLR;
	}

	Eigen::MatrixXd dPRMdphiLR(2, equation_size);
	dPRMdphiLR.setZero();
	if (abs(MR) < 1.0) {
		dPRMdphiLR = val(Ku * (MR - 1.0) * (1.0 + MR)) * dMRdphiLR;
	}


	Eigen::VectorXd dpFdphiL(equation_size);
	Eigen::VectorXd dpFdphiR(equation_size);

	{
		dpFdphiL[0] = val(
			0.5 * (dphiFLdphiLR(0, 0) + dphiFRdphiLR(0, 0)) -
			0.5 * (dPLPdphiLR(0, 0) - dPRMdphiLR(0, 0)) * (phiFLR(1, 0) - phiFLR(0, 0)) -
			0.5 * (PLP - PRM) * (dphiFRdphiLR(0, 0) - dphiFLdphiLR(0, 0)) +
			//dKLRdpL * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(0, 0) + dPRMdphiLR(0, 0)) * rhohat * chat +
			KLR * (PLP + PRM - 1.0) * 0.5 * (drhoFLdphiLR(0, 0) + drhoFRdphiLR(0, 0)) * chat +
			KLR * (PLP + PRM - 1.0) * rhohat * 0.5 * (dcFLdpL + dcFRdpL) -
			Ku * dPLPdphiLR(0, 0) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(0, 0) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(0, 0) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * 0.5 * (drhoFLdphiLR(0, 0) + drhoFRdphiLR(0, 0)) * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * 0.5 * (dcFLdpL + dcFRdpL) * (UnFR - UnFL));


		dpFdphiL[1] = val(-0.5 * (dPLPdphiLR(0, 1) - dPRMdphiLR(0, 1)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			dKLRduL * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(0, 1) + dPRMdphiLR(0, 1)) * rhohat * chat -
			Ku * dPLPdphiLR(0, 1) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(0, 1) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(0, 1) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * chat * (dUnFRduL - dUnFLduL));

		dpFdphiL[2] = val(-0.5 * (dPLPdphiLR(0, 2) - dPRMdphiLR(0, 2)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			dKLRdvL * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(0, 2) + dPRMdphiLR(0, 2)) * rhohat * chat -
			Ku * dPLPdphiLR(0, 2) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(0, 2) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(0, 2) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * chat * (dUnFRdvL - dUnFLdvL));

		dpFdphiL[3] = val(-0.5 * (dPLPdphiLR(0, 3) - dPRMdphiLR(0, 3)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			dKLRdwL * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(0, 3) + dPRMdphiLR(0, 3)) * rhohat * chat -
			Ku * dPLPdphiLR(0, 3) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(0, 3) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(0, 3) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * chat * (dUnFRdwL - dUnFLdwL));


		dpFdphiL[4] = val(
			-0.5 * (dPLPdphiLR(0, 4) - dPRMdphiLR(0, 4)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			KLR * (dPLPdphiLR(0, 4) + dPRMdphiLR(0, 4)) * rhohat * chat +
			KLR * (PLP + PRM - 1.0) * 0.5 * (drhoFLdphiLR(0, 4) + drhoFRdphiLR(0, 4)) * chat +
			KLR * (PLP + PRM - 1.0) * rhohat * 0.5 * (dcFLdTL + dcFRdTL) -
			Ku * dPLPdphiLR(0, 4) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(0, 4) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(0, 4) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * 0.5 * (drhoFLdphiLR(0, 4) + drhoFRdphiLR(0, 4)) * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * 0.5 * (dcFLdTL + dcFRdTL) * (UnFR - UnFL));

		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dpFdphiL[eq_str_spec_ + isp] = val(
				-0.5 * (dPLPdphiLR(0, eq_str_spec_ + isp) - dPRMdphiLR(0, eq_str_spec_ + isp)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
				KLR * (dPLPdphiLR(0, eq_str_spec_ + isp) + dPRMdphiLR(0, eq_str_spec_ + isp)) * rhohat * chat +
				KLR * (PLP + PRM - 1.0) * 0.5 * (drhoFLdphiLR(0, eq_str_spec_ + isp) + drhoFRdphiLR(0, eq_str_spec_ + isp)) * chat +
				KLR * (PLP + PRM - 1.0) * rhohat * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]) -
				Ku * dPLPdphiLR(0, eq_str_spec_ + isp) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
				Ku * PLP * dPRMdphiLR(0, eq_str_spec_ + isp) * fa * rhohat * chat * (UnFR - UnFL) -
				Ku * PLP * PRM * dfadphiLR(0, eq_str_spec_ + isp) * rhohat * chat * (UnFR - UnFL) -
				Ku * PLP * PRM * fa * 0.5 * (drhoFLdphiLR(0, eq_str_spec_ + isp) + drhoFRdphiLR(0, eq_str_spec_ + isp)) * chat * (UnFR - UnFL) -
				Ku * PLP * PRM * fa * rhohat * 0.5 * (dcFLdMFL[isp] + dcFRdMFL[isp]) * (UnFR - UnFL));
		}
	}



	{
		dpFdphiR[0] = val(
			0.5 * (dphiFLdphiLR(1, 0) + dphiFRdphiLR(1, 0)) -
			0.5 * (dPLPdphiLR(1, 0) - dPRMdphiLR(1, 0)) * (phiFLR(1, 0) - phiFLR(0, 0)) -
			0.5 * (PLP - PRM) * (dphiFRdphiLR(1, 0) - dphiFLdphiLR(1, 0)) +
			//dKLRdpL * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(1, 0) + dPRMdphiLR(1, 0)) * rhohat * chat +
			KLR * (PLP + PRM - 1.0) * 0.5 * (drhoFLdphiLR(1, 0) + drhoFRdphiLR(1, 0)) * chat +
			KLR * (PLP + PRM - 1.0) * rhohat * 0.5 * (dcFLdpR + dcFRdpR) -
			Ku * dPLPdphiLR(1, 0) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(1, 0) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(1, 0) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * 0.5 * (drhoFLdphiLR(1, 0) + drhoFRdphiLR(1, 0)) * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * 0.5 * (dcFLdpR + dcFRdpR) * (UnFR - UnFL));

		dpFdphiR[1] = val(-0.5 * (dPLPdphiLR(1, 1) - dPRMdphiLR(1, 1)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			dKLRduR * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(1, 1) + dPRMdphiLR(1, 1)) * rhohat * chat -
			Ku * dPLPdphiLR(1, 1) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(1, 1) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(1, 1) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * chat * (dUnFRduR - dUnFLduR));

		dpFdphiR[2] = val(-0.5 * (dPLPdphiLR(1, 2) - dPRMdphiLR(1, 2)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			dKLRdvR * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(1, 2) + dPRMdphiLR(1, 2)) * rhohat * chat -
			Ku * dPLPdphiLR(1, 2) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(1, 2) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(1, 2) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * chat * (dUnFRdvR - dUnFLdvR));

		dpFdphiR[3] = val(-0.5 * (dPLPdphiLR(1, 3) - dPRMdphiLR(1, 3)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			dKLRdwR * (PLP + PRM - 1.0) * rhohat * chat +
			KLR * (dPLPdphiLR(1, 3) + dPRMdphiLR(1, 3)) * rhohat * chat -
			Ku * dPLPdphiLR(1, 3) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(1, 3) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(1, 3) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * chat * (dUnFRdwR - dUnFLdwR));


		dpFdphiR[4] = val(
			-0.5 * (dPLPdphiLR(1, 4) - dPRMdphiLR(1, 4)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
			KLR * (dPLPdphiLR(1, 4) + dPRMdphiLR(1, 4)) * rhohat * chat +
			KLR * (PLP + PRM - 1.0) * 0.5 * (drhoFLdphiLR(1, 4) + drhoFRdphiLR(1, 4)) * chat +
			KLR * (PLP + PRM - 1.0) * rhohat * 0.5 * (dcFLdTR + dcFRdTR) -
			Ku * dPLPdphiLR(1, 4) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * dPRMdphiLR(1, 4) * fa * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * dfadphiLR(1, 4) * rhohat * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * 0.5 * (drhoFLdphiLR(1, 4) + drhoFRdphiLR(1, 4)) * chat * (UnFR - UnFL) -
			Ku * PLP * PRM * fa * rhohat * 0.5 * (dcFLdTR + dcFRdTR) * (UnFR - UnFL));

		for (int isp = 0; isp < eq_size_spec_; ++isp) {
			dpFdphiR[eq_str_spec_ + isp] = val(
				-0.5 * (dPLPdphiLR(1, eq_str_spec_ + isp) - dPRMdphiLR(1, eq_str_spec_ + isp)) * (phiFLR(1, 0) - phiFLR(0, 0)) +
				KLR * (dPLPdphiLR(1, eq_str_spec_ + isp) + dPRMdphiLR(1, eq_str_spec_ + isp)) * rhohat * chat +
				KLR * (PLP + PRM - 1.0) * 0.5 * (drhoFLdphiLR(1, eq_str_spec_ + isp) + drhoFRdphiLR(1, eq_str_spec_ + isp)) * chat +
				KLR * (PLP + PRM - 1.0) * rhohat * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]) -
				Ku * dPLPdphiLR(1, eq_str_spec_ + isp) * PRM * fa * rhohat * chat * (UnFR - UnFL) -
				Ku * PLP * dPRMdphiLR(1, eq_str_spec_ + isp) * fa * rhohat * chat * (UnFR - UnFL) -
				Ku * PLP * PRM * dfadphiLR(1, eq_str_spec_ + isp) * rhohat * chat * (UnFR - UnFL) -
				Ku * PLP * PRM * fa * 0.5 * (drhoFLdphiLR(1, eq_str_spec_ + isp) + drhoFRdphiLR(1, eq_str_spec_ + isp)) * chat * (UnFR - UnFL) -
				Ku * PLP * PRM * fa * rhohat * 0.5 * (dcFLdMFR[isp] + dcFRdMFR[isp]) * (UnFR - UnFL));
		}
	}


	Eigen::MatrixXd dfluxdphiL(equation_size, equation_size);
	Eigen::MatrixXd dfluxdphiR(equation_size, equation_size);

	{
		dfluxdphiL.row(0) = dmdotdphiL;
		if (mdot > 0.0) {
			dfluxdphiL.row(1) = dmdotdphiL * val(phiFLR(0, 1)) + nvec[0] * dpFdphiL;
			dfluxdphiL.coeffRef(1, 1) += val(mdot * dphiFLdphiLR(0, 1));
			dfluxdphiL.row(2) = dmdotdphiL * val(phiFLR(0, 2)) + nvec[1] * dpFdphiL;
			dfluxdphiL.coeffRef(2, 2) += val(mdot * dphiFLdphiLR(0, 2));
			dfluxdphiL.row(3) = dmdotdphiL * val(phiFLR(0, 3)) + nvec[2] * dpFdphiL;
			dfluxdphiL.coeffRef(3, 3) += val(mdot * dphiFLdphiLR(0, 3));
			dfluxdphiL.row(4) = dmdotdphiL.transpose() * val(HtFL) + val(mdot) * dHtFLdphiLR.row(0);
			dfluxdphiL.row(5) = dmdotdphiL * val(phiFLR(0, 5));
			dfluxdphiL.coeffRef(5, 5) += val(mdot * dphiFLdphiLR(0, 5));
		}
		else {
			dfluxdphiL.row(1) = dmdotdphiL * val(phiFLR(1, 1)) + nvec[0] * dpFdphiL;
			dfluxdphiL.coeffRef(1, 1) += val(mdot * dphiFRdphiLR(0, 1));
			dfluxdphiL.row(2) = dmdotdphiL * val(phiFLR(1, 2)) + nvec[1] * dpFdphiL;
			dfluxdphiL.coeffRef(2, 2) += val(mdot * dphiFRdphiLR(0, 2));
			dfluxdphiL.row(3) = dmdotdphiL * val(phiFLR(1, 3)) + nvec[2] * dpFdphiL;
			dfluxdphiL.coeffRef(3, 3) += val(mdot * dphiFRdphiLR(0, 3));
			dfluxdphiL.row(4) = dmdotdphiL.transpose() * val(HtFR) + val(mdot) * dHtFRdphiLR.row(0);
			dfluxdphiL.row(5) = dmdotdphiL * val(phiFLR(1, 5));
			dfluxdphiL.coeffRef(5, 5) += val(mdot * dphiFRdphiLR(0, 5));
		}
	}


	{
		dfluxdphiR.row(0) = dmdotdphiR;
		if (mdot > 0.0) {
			dfluxdphiR.row(1) = dmdotdphiR * val(phiFLR(0, 1)) + nvec[0] * dpFdphiR;
			dfluxdphiR.coeffRef(1, 1) += val(mdot * dphiFLdphiLR(1, 1));
			dfluxdphiR.row(2) = dmdotdphiR * val(phiFLR(0, 2)) + nvec[1] * dpFdphiR;
			dfluxdphiR.coeffRef(2, 2) += val(mdot * dphiFLdphiLR(1, 2));
			dfluxdphiR.row(3) = dmdotdphiR * val(phiFLR(0, 3)) + nvec[2] * dpFdphiR;
			dfluxdphiR.coeffRef(3, 3) += val(mdot * dphiFLdphiLR(1, 3));
			dfluxdphiR.row(4) = dmdotdphiR.transpose() * val(HtFL) + val(mdot) * dHtFLdphiLR.row(1);
			dfluxdphiR.row(5) = dmdotdphiR * val(phiFLR(0, 5));
			dfluxdphiR.coeffRef(5, 5) += val(mdot * dphiFLdphiLR(1, 5));
		}
		else {
			dfluxdphiR.row(1) = dmdotdphiR * val(phiFLR(1, 1)) + nvec[0] * dpFdphiR;
			dfluxdphiR.coeffRef(1, 1) += val(mdot * dphiFRdphiLR(1, 1));
			dfluxdphiR.row(2) = dmdotdphiR * val(phiFLR(1, 2)) + nvec[1] * dpFdphiR;
			dfluxdphiR.coeffRef(2, 2) += val(mdot * dphiFRdphiLR(1, 2));
			dfluxdphiR.row(3) = dmdotdphiR * val(phiFLR(1, 3)) + nvec[2] * dpFdphiR;
			dfluxdphiR.coeffRef(3, 3) += val(mdot * dphiFRdphiLR(1, 3));
			dfluxdphiR.row(4) = dmdotdphiR.transpose() * val(HtFR) + val(mdot) * dHtFRdphiLR.row(1);
			dfluxdphiR.row(5) = dmdotdphiR * val(phiFLR(1, 5));
			dfluxdphiR.coeffRef(5, 5) += val(mdot * dphiFRdphiLR(1, 5));
		}
	}


	return std::make_pair(dfluxdphiL, dfluxdphiR);
}






int main() {

	int species_size = 2;
	int eq_size_spec_ = 1;
	int eq_str_spec_ = 5;
	int equation_size = 5 + eq_size_spec_;

	autodiff::VectorXvar W(2 * equation_size);

	auto& pL = W[equation_size * 0 + 0];
	auto& uL = W[equation_size * 0 + 1];
	auto& vL = W[equation_size * 0 + 2];
	auto& wL = W[equation_size * 0 + 3];
	auto& TL = W[equation_size * 0 + 4];
	auto& Y0L = W[equation_size * 0 + 5];

	auto& pR = W[equation_size * 1 + 0];
	auto& uR = W[equation_size * 1 + 1];
	auto& vR = W[equation_size * 1 + 2];
	auto& wR = W[equation_size * 1 + 3];
	auto& TR = W[equation_size * 1 + 4];
	auto& Y0R = W[equation_size * 1 + 5];

	pL = 601325.0;
	uL = -7.0;
	vL = 2.0;
	wL = 3.0;
	TL = 300.0;
	Y0L = 0.4;


	pR = 501325.0;
	uR = 5.0;
	vR = -2.0;
	wR = -1.0;
	TR = 320.0;
	Y0R = 0.6;

	Eigen::MatrixXd DWF_phiLR(2, equation_size);

	DWF_phiLR.row(0) << 0.1, 0.2, 0.3, 0.4, 0.7, 0.6;
	DWF_phiLR.row(1) << 0.6, 0.7, 0.4, 0.3, 0.2, 0.1;

	autodiff::MatrixXvar phiFLR(2, equation_size);

	for (int ieq = 0; ieq < equation_size; ++ieq) {
		phiFLR(0, ieq) =
			(1.0 - DWF_phiLR(0, ieq)) * W[ieq] + DWF_phiLR(0, ieq) * W[equation_size + ieq];
		phiFLR(1, ieq) = 
			(1.0 - DWF_phiLR(1, ieq)) * W[equation_size + ieq] + DWF_phiLR(1, ieq) * W[ieq];
	}


	using EOS_T = std::variant<EOS_ideal<autodiff::var>, EOS_NASG<autodiff::var>>;
	std::vector<EOS_T> eoses;

	eoses.push_back(EOS_ideal<autodiff::var>(1.4, 717.5));
	eoses.push_back(EOS_NASG<autodiff::var>(1.19, 3610.0, 6.2178E8, 6.7212E-4, -1.177788E6));


	autodiff::var rhoFL, HtFL, cFL, dHtFLdpFL, dHtFLdTFL, drhoFLdpFL, drhoFLdTFL, drhoFLdpdpFL, drhoFLdpdTFL, drhoFLdTdTFL;
	std::vector<autodiff::var> drhoFLdMFFL(species_size);
	std::vector<autodiff::var> dHtFLdMFFL(species_size);
	std::vector<autodiff::var> drhoFLdpdMFFL(species_size);
	std::vector<autodiff::var> drhoFLdTdMFFL(species_size);
	std::vector<autodiff::var> dHtdpdMFFL(species_size);
	std::vector<autodiff::var> dHtdTdMFFL(species_size);
	mixture_thermodynamic_properties(
		species_size, eoses,
		phiFLR(0, 0), phiFLR(0, 1), phiFLR(0, 2), phiFLR(0, 3), phiFLR(0, 4), { phiFLR(0, 5), 1.0 - phiFLR(0, 5) },
		rhoFL,
		HtFL,
		cFL,
		dHtFLdpFL,
		dHtFLdTFL,
		drhoFLdpFL,
		drhoFLdTFL,
		drhoFLdpdpFL,
		drhoFLdpdTFL,
		drhoFLdTdTFL,
		drhoFLdMFFL,
		dHtFLdMFFL,
		drhoFLdpdMFFL,
		drhoFLdTdMFFL,
		dHtdpdMFFL,
		dHtdTdMFFL
	);

	autodiff::var rhoFR, HtFR, cFR, dHtFRdpFR, dHtFRdTFR, drhoFRdpFR, drhoFRdTFR, drhoFRdpdpFR, drhoFRdpdTFR, drhoFRdTdTFR;
	std::vector<autodiff::var> drhoFRdMFFR(species_size);
	std::vector<autodiff::var> dHtFRdMFFR(species_size);
	std::vector<autodiff::var> drhoFRdpdMFFR(species_size);
	std::vector<autodiff::var> drhoFRdTdMFFR(species_size);
	std::vector<autodiff::var> dHtdpdMFFR(species_size);
	std::vector<autodiff::var> dHtdTdMFFR(species_size);
	mixture_thermodynamic_properties(
		species_size, eoses,
		phiFLR(1, 0), phiFLR(1, 1), phiFLR(1, 2), phiFLR(1, 3), phiFLR(1, 4), { phiFLR(1, 5), 1.0 - phiFLR(1, 5) },
		rhoFR,
		HtFR,
		cFR,
		dHtFRdpFR,
		dHtFRdTFR,
		drhoFRdpFR,
		drhoFRdTFR,
		drhoFRdpdpFR,
		drhoFRdpdTFR,
		drhoFRdTdTFR,
		drhoFRdMFFR,
		dHtFRdMFFR,
		drhoFRdpdMFFR,
		drhoFRdTdMFFR,
		dHtdpdMFFR,
		dHtdTdMFFR
	);

	Eigen::Vector3d nvec = { 1.0, 3.0, 6.0 };
	nvec.normalize();


	std::cout << std::endl;
	std::cout << autodiff::gradient(flux0, W).transpose() << std::endl;
	std::cout << autodiff::gradient(flux1, W).transpose() << std::endl;
	std::cout << autodiff::gradient(flux2, W).transpose() << std::endl;
	std::cout << autodiff::gradient(flux3, W).transpose() << std::endl;
	std::cout << autodiff::gradient(flux4, W).transpose() << std::endl;
	std::cout << autodiff::gradient(flux5, W).transpose() << std::endl;
	std::cout << std::endl;

	std::cout << dfluxdphiL << std::endl << std::endl;
	std::cout << dfluxdphiR << std::endl << std::endl;




}