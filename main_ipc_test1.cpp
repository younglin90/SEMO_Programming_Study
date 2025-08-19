

#include <Eigen/Dense>

//수도코드
// project a symmetric real matrix to the nearest SPD matrix
template <typename Scalar, int size>
static void makePD(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if (eigenSolver.eigenvalues()[0] >= 0.0) {
        return;
    }
    Eigen::DiagonalMatrix<Scalar, size> D(eigenSolver.eigenvalues());
    int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for (int i = 0; i < rows; i++) {
        if (D.diagonal()[i] < 0.0) {
            D.diagonal()[i] = 0.0;
        }
        else {
            break;
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

int main() {


int time_integral_method = [0, 1] ; backward_euler, Newmark
bool is_calc_gravity = false;

backward_euler :
dx = xk - xhat;
vk = (xk - xkm) / dt;
xkm = xk;

Newmark :
dx = xk - xhat;
vk = 2.0 * (xk - xkm) / dt - vkm;
ak = 4.0 * (xk - xhat) / dt / dt + g;
xkm = xk;



enum class TimeIntegralMethod {
    BackwardEuler, Newmark
};

TimeIntegralMethod tim;
tim = TimeIntegralMethod::BackwardEuler;



Eigen::VectorXd searchDir;
Eigen::VectorXd position, position0;
Eigen::VectorXd velocity;
Eigen::VectorXd acceleration;

double dt, stepSize;
Eigen::Vector3d gravity;

// 초기 탐색 방향 설정
auto init_x = [&]() {

    searchDir.conservativeResize(position.rows());

    if (tim == TimeIntegralMethod::BackwardEuler) {
        searchDir = dt * velocity;// +dt * dt * gravity;
    }
    else {
        searchDir = dt * velocity;// +0.5 * dt * dt * gravity;
    }

    };


// 메시의 정점 위치를 탐색 방향(search direction)을 따라 업데이트
auto step_forward = [&]() {

    position = position0 + stepSize * searchDir;

    };

// 제약조건 계산
std::vector<std::pair<size_t, size_t>> constraintSetVT, constraintSetTV, constraintSetEE;
auto compute_constraint_sets = [&]() {


    //// 제약조건 세트 추가
    //// vertex - tri
    //if (d < dhat) {
    //    constraintSetVT[sfI].push_back(-vI - 1, sfVInd[0], -1, -1);
    //}
    //// tri-vertex
    //if (d < dhat) {
    //    constraintSetTV[sfI].push_back(-vI - 1, sfVInd[0], -1, -1);
    //}
    //// edge-edge
    //if (d < dhat) {
    //    constraintSetEE[sfI].push_back(-vI - 1, sfVInd[0], -1, -1);
    //}

    //constraintSet < -PT, TP, EE;

    };


// b 헤시안
auto compute_H_b = [&](double d1, double dHat1, double& H) {
    double t2 = d1 - dHat1;
    return (std::log(d1 / dHat1) * -2.0 - t2 * 4.0 / d1) + 1.0 / (d1 * d1) * (t2 * t2);
    };

// b 그래드
auto compute_g_b = [&](double d1, double dHat1, double& H) {
    double t2 = d1 - dHat1;
    return t2 * std::log(d1 / dHat1) * -2.0 - (t2 * t2) / d1;
    };

// b
auto compute_b = [&](double d1, double dHat1, double& H) {
    return -(d1 - dHat1) * (d1 - dHat1) * log(d1 / dHat1);
    };




auto upperBoundMu = [&](double& mu)
    {
        double H_b;
        compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
        double muMax = 1.0e13 * result.avgNodeMass(dim) / (4.0e-16 * bboxDiagSize2 * H_b);
        if (mu > muMax) mu = muMax;
    };



auto suggestMu = [&](double& mu)
{
    double H_b;
    compute_H_b(1.0e-16 * bboxDiagSize2, dHat, H_b);
    mu = 1.0e11 * result.avgNodeMass(dim) / (4.0e-16 * bboxDiagSize2 * H_b);
}



// 어댑티브 mu 구하기 (복잡함)
initMu_IP


evaluateConstraints
evaluateConstraint
// edge-edge
// vertex-tri
// tri-vertex
val = d;// d = 거리
val *= coef; // coef 기본값 1.0

constraintVal <- val;// 여기서 구한 val 을 저장






computeEnergyVal
lastEnergyVal = ;




while (true) {

    //initSubProb_IP();
    closeConstraintID.resize(0);
    closeMConstraintID.resize(0);
    closeConstraintVal.resize(0);
    closeMConstraintVal.resize(0);




    //solveSub_IP(mu_IP, activeSet_next, MMActiveSet_next);
    int iterCap = 10000, k = 0;
    //m_projectDBC = true; // 드리트리 바운더리 조건
    //rho_DBC = 0.0;
    double lastMove = animScripter.stepSize;
    for (; k < iterCap; ++k) {
        //buildConstraintStartIndsWithMM(
        //    activeSet, MMActiveSet, constraintStartInds);
        constraintStartInds.resize(1);
        constraintStartInds[0] = 0;
        for (int coI = 0; coI < animConfig.collisionObjects.size(); ++coI) {
            constraintStartInds.emplace_back(constraintStartInds.back() + activeSet[coI].size());
        }
        for (int coI = 0; coI < animConfig.meshCollisionObjects.size(); ++coI) {
            constraintStartInds.emplace_back(constraintStartInds.back() + MMActiveSet[coI].size());
        }
        if (animConfig.isSelfCollision) {
            constraintStartInds.emplace_back(constraintStartInds.back() + MMActiveSet.back().size());
        }



        //computeGradient(result, false, gradient, m_projectDBC);
        if (tim == TimeIntegralMethod::BackwardEuler) {
            computeGradient();
        }
        else {
            computeGradient();
        }

        gradient.segment<dim>(vI* dim) += 
            (data.massMatrix.coeff(vI, vI) * (data.V.row(vI) - 
                xTilta.row(vI)).transpose());
        evaluateConstraints
        evaluateConstraint
        // edge-edge
        // vertex-tri
        // tri-vertex
        val = d;// d = 거리
        val *= coef; // coef 기본값 1.0

        constraintVal <- val;// 여기서 구한 val 을 저장

        constraintVal <- compute_g_b;

        //leftMultiplyConstraintJacobianT;
        // edge-edge
        // vertex-tri
        // tri-vertex
        gradient += g * coef * -MMCVIDI[3] * input[constraintI];


        //augmentParaEEGradient; -> edge-edge eps_x, e, e_g 등으로 계산



        // check convergence
        double gradSqNorm = gradient.squaredNorm();
        double distToOpt_PN = searchDir.cwiseAbs().maxCoeff();
        bool gradVanish = (distToOpt_PN < targetGRes);
        if (!useGD && k && gradVanish && (animScripter.getCompletedStepSize() > 1.0 - 1.0e-3)
            && (animScripter.getCOCompletedStepSize() > 1.0 - 1.0e-3)) {
            break;
        }


        //computeSearchDir(k, m_projectDBC);
        //computePrecondMtr(result, false, linSysSolver, false, projectDBC);
        p_linSysSolver->setZero();
        // backward_euler:
        energyTerms[eI]->computeHessian(data, redoSVD, svd, F,
            energyParams[eI] * dtSq,
            p_linSysSolver, true, projectDBC);
        // newmark:
        energyTerms[eI]->computeHessian(data, redoSVD, svd, F, 
            energyParams[eI] * dtSq* beta_NM, 
            p_linSysSolver, true, projectDBC);

        double massI = data.massMatrix.coeff(vI, vI);
        int ind0 = vI * dim;
        int ind1 = ind0 + 1;
        p_linSysSolver->addCoeff(ind0, ind0, massI);
        p_linSysSolver->addCoeff(ind1, ind1, massI);
        p_linSysSolver->addCoeff(ind2, ind2, massI);

        //animConfig.meshCollisionObjects[coI]->
        //    augmentIPHessian(data, MMActiveSet[coI], 
        //        p_linSysSolver, dHat, kappa, projectDBC);
        std::vector<Eigen::Matrix<double, 4 * dim, 4 * dim>> IPHessian(activeSet.size());
        std::vector<Eigen::Matrix<int, 4, 1>> rowIStart(activeSet.size());
        // edge-edge
        double d;
        IPHessian[cI] = ((coef * H_b) * g) * g.transpose() + 
            (coef * g_b) * H;
        IglUtils::makePD(IPHessian[cI]);
        // point-triangle and degenerate edge-edge
        // PP, PE
        double coef_dup = coef * -MMCVIDI[3];
        Eigen::Matrix<double, dim * 2, dim * 2> HessianBlock =
            ((coef_dup * H_b) * g) * g.transpose() + 
            (coef_dup * g_b) * H;
        IglUtils::makePD(HessianBlock);
        // PT
        IPHessian[cI] = ((coef * H_b) * g) * g.transpose() + 
            (coef * g_b) * H;
        IglUtils::makePD(IPHessian[cI]);
        // triangle-point
        IPHessian[cI] = ((coef * H_b) * g) * g.transpose() + 
            (coef * g_b) * H;
        IglUtils::makePD(IPHessian[cI]);
        // edge-point
        double coef_dup = coef * -MMCVIDI[3];
        Eigen::Matrix<double, dim * 3, dim * 3> HessianBlock = 
            ((coef_dup * H_b) * g) * g.transpose() + 
            (coef_dup * g_b) * H;
        IglUtils::makePD(HessianBlock);





        //animConfig.meshCollisionObjects[coI]->
        //    augmentParaEEHessian(data, paraEEMMCVIDSet[coI], 
        //        paraEEeIeJSet[coI],
        //    p_linSysSolver, dHat, kappa, projectDBC);
        // eps_x, e, e_g, e_H 계산
        // PE, EP, PP, 계산
        Eigen::Matrix<double, 12, 12> kappa_gradb_gradeT;
        kappa_gradb_gradeT = 
            ((coef * g_b) * grad_d) * e_g.transpose();
        PEEHessian[cI] = kappa_gradb_gradeT + 
            kappa_gradb_gradeT.transpose() + 
            (coef * b) * e_H + 
            ((coef * e * H_b) * grad_d) * grad_d.transpose() +
            (coef * e * g_b) * H_d;
        IglUtils::makePD(PEEHessian[cI]);


        // solve
        Eigen::VectorXd minusG = -gradient;
        linSysSolver->solve(minusG, searchDir);




        //double alpha = 1.0, slackness_a = 0.9, slackness_m = 0.8;
        //energyTerms[0]->filterStepSize(result, searchDir, alpha);
        //for (int coI = 0; coI < animConfig.collisionObjects.size(); ++coI) {
        //    animConfig.collisionObjects[coI]->
        //        largestFeasibleStepSize(
        //            result, searchDir, slackness_a, 
        //            AHat[coI], alpha);
        //}


        sh.build(result, searchDir, alpha, result.avgEdgeLen / 3.0);

        // full CCD or partial CCD
        largestFeasibleStepSize_TightInclusion(
            result, sh, searchDir, 
            animConfig.ccdTolerance, 
            MMActiveSet_CCD[coI], alpha);



        std::vector<std::pair<int, int>> newCandidates;


        double alpha_feasible = alpha;




        //lineSearch(alpha);
        computeEnergyVal(result, false, lastEnergyVal);
        double c1m = 0.0;
        if (armijoParam > 0.0) {
            c1m = armijoParam * searchDir.dot(gradient);
        }
        Eigen::MatrixXd resultV0 = result.V;
        stepForward(resultV0, result, stepSize);
        if (energyTerms[0]->getNeedElemInvSafeGuard()) {
            while (!result.checkInversion(true)) {
                stepSize /= 2.0;
                stepForward(resultV0, result, stepSize);
            }
        }
        sh.build(result, result.avgEdgeLen / 3.0);
        while (isIntersected(result, sh, resultV0, animConfig)) {
            stepSize /= 2.0;
            stepForward(resultV0, result, stepSize);
            sh.build(result, result.avgEdgeLen / 3.0);
        }
        computeConstraintSets(result, rehash);
        computeEnergyVal(result, 2, testingE);
        double LFStepSize = stepSize;
        while ((testingE > lastEnergyVal + stepSize * c1m) && // Armijo condition
            (stepSize > lowerBound)) {
            // fprintf(out, "%.9le %.9le\n", stepSize, testingE);
            if (stepSize == 1.0) {
                // can try cubic interpolation here
                stepSize /= 2.0;
            }
            else {
                stepSize /= 2.0;
            }

            ++numOfLineSearch;
            if (stepSize == 0.0) {
                stopped = true;
                break;
            }
            stepForward(resultV0, result, stepSize);
            computeConstraintSets(result);
            computeEnergyVal(result, 2, testingE);
        }

        if (stepSize < LFStepSize) {
            bool needRecomputeCS = false;
            while (isIntersected(result, sh, resultV0, animConfig)) {

                stepSize /= 2.0;
                stepForward(resultV0, result, stepSize);
                sh.build(result, result.avgEdgeLen / 3.0);
                needRecomputeCS = true;
            }
        }






        // adaptive kappa 구하는 부분
        postLineSearch(alpha);



    }




    constraintVal = evaluateConstraints;




    if (!constraintVal.empty()) {

        fb(constraintVal.size());

        compute_g_b(constraintVal, dHat, dualI);
        dualI *= -mu_IP;
        fb = (dualI + constraint -
            std::sqrt(dualI * dualI + constraint * constraint));

        fbNorm = fb.norm();

        if (fbNorm < fbNormTol || constraintVal.minCoeff() < dTol)
            break;


        if (HOMOTOPY_VAR == 0) {
            mu_IP *= 0.5;
        }
        else if (HOMOTOPY_VAR == 1) {
            if (updateDHat) {
                dHat *= 0.5;
                if (dHat < DHatTarget) dHat = dHatTarget;

                computeConstraintSets;

                initMu_IP(mu_IP);
            }
        }


    }


    //computeSystemEnergy(sysE, sysM, sysL);
    Eigen::VectorXd energyValPerElem;
    energyTerms[0]->getEnergyValPerElemBySVD(result, true, svd, F, energyValPerElem, false);

    sysE.resize(compVAccSize.size());
    sysM.resize(compVAccSize.size());
    sysL.resize(compVAccSize.size());
    for (int compI = 0; compI < compVAccSize.size(); ++compI) {
        sysE[compI] = 0.0;
        sysM[compI].setZero();
        sysL[compI].setZero();

        for (int fI = (compI ? compFAccSize[compI - 1] : 0); fI < compFAccSize[compI]; ++fI) {
            sysE[compI] += energyValPerElem[fI];
        }
        for (int vI = (compI ? compVAccSize[compI - 1] : 0); vI < compVAccSize[compI]; ++vI) {
            sysE[compI] += result.massMatrix.coeff(vI, vI) * ((result.V.row(vI) - result.V_prev.row(vI)).squaredNorm() / dtSq / 2.0 - gravity.dot(result.V.row(vI).transpose()));

            Eigen::Matrix<double, 1, dim> p = result.massMatrix.coeff(vI, vI) / dt * (result.V.row(vI) - result.V_prev.row(vI));
            sysM[compI] += p;

            if constexpr (dim == 3) {
                sysL[compI] += Eigen::Matrix<double, 1, dim>(result.V.row(vI)).cross(p);
            }
            else {
                sysL[compI][0] += result.V(vI, 0) * p[1] - result.V(vI, 1) * p[0];
            }
        }
    }


}




// backward_euler
dx_elastic = V - xTilta;
velocity = (V - V_prev) / dt;
//computeXTilta();
xTilta.conservativeResize(result.V.rows(), dim);
xTilta.row(vI) = (result.V_prev.row(vI) + 
    (velocity.segment<dim>(vI * dim) * dt + 
        gravityDtSq).transpose());

// newmark
dx_Elastic = V - xTilta;
velocity = 2.0 * (V - V_prev) / dt - velocity;
acceleration = (V - xTilta) * 4.0 / dtSq + gravity;
V_prev = V;
//computeXTilta();
xTilta.conservativeResize(result.V.rows(), dim);
xTilta.row(vI) = (result.V_prev.row(vI) + 
    (velocity.segment<dim>(vI * dim) * dt + 
        beta_NM * gravityDtSq + 
        (0.5 - beta_NM) * 
        (dtSq * acceleration.row(vI).transpose())).transpose());




















}