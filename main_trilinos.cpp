

// compare_solvers_tpetra_shylu.cpp
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <chrono>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)
#endif


// Galeri::Xpetra
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraMatrixTypes.hpp>
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>

// Thyra includes
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>
#include <Thyra_DefaultLinearOpSource.hpp>


// Stratimikos includes
#include <Stratimikos_FROSch_def.hpp>
#include <Stratimikos_LinearSolverBuilder.hpp>
#include <Stratimikos_MueLuHelpers.hpp>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>

// Tpetra includes
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>

// Xpetra include
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>

// FROSCH thyra includes
#include <Thyra_FROSchLinearOp_def.hpp>
#include <Thyra_FROSchFactory_def.hpp>
#include <FROSch_Tools_def.hpp>

// Teuchos includes
#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_GlobalMPISession.hpp>

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>

// Amesos2 includes
#include <Amesos2.hpp>

// Ifpack2 includes
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>

// MueLu includes
#include <MueLu_CreateTpetraPreconditioner.hpp>



#ifdef _MSC_VER
#pragma warning(pop)
#endif



using UN = unsigned;
using SC = double;
using LO = int;
using GO = FROSch::DefaultGlobalOrdinal;
using NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;
using namespace Thyra;

int main(int argc, char* argv[])
{
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc, &argv, &blackhole);

    RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();

    CommandLineProcessor My_CLP;

    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

    int M = 3;
    My_CLP.setOption("M", &M, "H / h.");
    int Dimension = 3;
    My_CLP.setOption("DIM", &Dimension, "Dimension.");
    int Overlap = 0;
    My_CLP.setOption("O", &Overlap, "Overlap.");
    string xmlFile = "ParameterList.xml";
    My_CLP.setOption("PLIST", &xmlFile, "File name of the parameter list.");
    bool useepetra = false;
    My_CLP.setOption("USEEPETRA", "USETPETRA", &useepetra, "Use Epetra infrastructure for the linear algebra.");
    bool useGeoMap = false;
    My_CLP.setOption("useGeoMap", "useAlgMap", &useGeoMap, "Use Geometric Map");
    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc, argv);
    if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }


    CommWorld->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Thyra Elasticity Test"));
    TimeMonitor::setStackedTimer(stackedTimer);

    int N = 0;
    int color = 1;
    if (Dimension == 2) {
        N = (int)(pow(CommWorld->getSize(), 1 / 2.) + 100 * numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank() < N * N) {
            color = 0;
        }
    }
    else if (Dimension == 3) {
        N = (int)(pow(CommWorld->getSize(), 1 / 3.) + 100 * numeric_limits<double>::epsilon()); // 1/H
        if (CommWorld->getRank() < N * N * N) {
            color = 0;
        }
    }
    else {
        assert(false);
    }

    UnderlyingLib xpetraLib = UseTpetra;
    if (useepetra) {
        xpetraLib = UseEpetra;
    }
    else {
        xpetraLib = UseTpetra;
    }

    RCP<const Comm<int> > Comm = CommWorld->split(color, CommWorld->getRank());


    if (color == 0) {

        //RCP<ParameterList> parameterList = getParametersFromXmlFile(xmlFile);
        RCP<ParameterList> parameterList = rcp(new ParameterList("Thyra Example"));

        // === 상위 선택 ===
        parameterList->set("Linear Solver Type", "Belos");
        parameterList->set("Preconditioner Type", "FROSch");

        // === Belos 설정 ===
        {
            auto& lsTypes = parameterList->sublist("Linear Solver Types");
            auto& belos = lsTypes.sublist("Belos");
            belos.set("Solver Type", "Block GMRES");

            auto& solverTypes = belos.sublist("Solver Types");
            auto& blockGMRES = solverTypes.sublist("Block GMRES");
            blockGMRES.set("PreconditionerPosition", "left");
            blockGMRES.set("Block Size", 1);
            blockGMRES.set("Convergence Tolerance", 1e-4);
            blockGMRES.set("Maximum Iterations", 100);
            blockGMRES.set("Output Frequency", 1);
            blockGMRES.set("Show Maximum Residual Norm Only", true);
        }

        // === FROSch 설정 ===
        {
            auto& precTypes = parameterList->sublist("Preconditioner Types");
            auto& frosch = precTypes.sublist("FROSch");

            // 최상위 FROSch 키
            frosch.set("FROSch Preconditioner Type", "GDSWPreconditioner");

            // AlgebraicOverlappingOperator
            {
                auto& aoo = frosch.sublist("AlgebraicOverlappingOperator");
                auto& solver = aoo.sublist("Solver");
                solver.set("SolverType", "Ifpack2");
                solver.set("Solver", "RILUK");

                auto& ifp = solver.sublist("Ifpack2");
                ifp.set("fact: iluk level-of-fill", 3);
                // Ifpack2에서 HTS 삼각해를 강제하려면(원하실 경우):
                // ifp.set("trisolver: type", "HTS");
            }

            // GDSWCoarseOperator
            {
                auto& gdsw = frosch.sublist("GDSWCoarseOperator");

                // Blocks / 1
                {
                    auto& blocks = gdsw.sublist("Blocks");
                    auto& b1 = blocks.sublist("1");
                    b1.set("Use For Coarse Space", true);
                    b1.set("Rotations", true);
                }

                // ExtensionSolver
                {
                    auto& ext = gdsw.sublist("ExtensionSolver");
                    ext.set("SolverType", "Amesos2");
                    ext.set("Solver", "Klu");   // 보통은 "KLU2"를 많이 사용합니다(빌드에 따라 확인 권장).
                }

                // Distribution
                {
                    auto& dist = gdsw.sublist("Distribution");
                    dist.set("Type", "linear");
                    dist.set("NumProcs", 1);
                    dist.set("Factor", 1.0);
                    dist.set("GatheringSteps", 1);

                    auto& comm = dist.sublist("Gathering Communication");
                    comm.set("Send type", "Send");
                }

                // CoarseSolver
                {
                    auto& coarse = gdsw.sublist("CoarseSolver");
                    coarse.set("SolverType", "Amesos2");
                    coarse.set("Solver", "Klu"); // 필요시 "KLU2"로 교체
                }
            }
        }

        Comm->barrier();
        if (Comm->getRank() == 0) {
            cout << "##################\n# Parameter List #\n##################" << endl;
            parameterList->print(cout);
            cout << endl;
        }

        Comm->barrier(); if (Comm->getRank() == 0) cout << "##############################\n# Assembly Laplacian #\n##############################\n" << endl;

        ParameterList GaleriList;
        GaleriList.set("nx", GO(N * M));
        GaleriList.set("ny", GO(N * M));
        GaleriList.set("nz", GO(N * M));
        GaleriList.set("mx", GO(N));
        GaleriList.set("my", GO(N));
        GaleriList.set("mz", GO(N));


        RCP<const Map<LO, GO, NO> > UniqueNodeMap;
        RCP<const Map<LO, GO, NO> > UniqueMap;
        RCP<MultiVector<SC, LO, GO, NO> > Coordinates;
        RCP<Matrix<SC, LO, GO, NO> > K;
        if (Dimension == 2) {
            UniqueNodeMap = Galeri::Xpetra::CreateMap<LO, GO, NO>(xpetraLib, "Cartesian2D", Comm, GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
            UniqueMap = Xpetra::MapFactory<LO, GO, NO>::Build(UniqueNodeMap, 2);
            Coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC, LO, GO, Map<LO, GO, NO>, MultiVector<SC, LO, GO, NO> >("2D", UniqueMap, GaleriList);
            RCP<Galeri::Xpetra::Problem<Map<LO, GO, NO>, CrsMatrixWrap<SC, LO, GO, NO>, MultiVector<SC, LO, GO, NO> > > Problem = Galeri::Xpetra::BuildProblem<SC, LO, GO, Map<LO, GO, NO>, CrsMatrixWrap<SC, LO, GO, NO>, MultiVector<SC, LO, GO, NO> >("Elasticity2D", UniqueMap, GaleriList);
            K = Problem->BuildMatrix();
        }
        else if (Dimension == 3) {
            UniqueNodeMap = Galeri::Xpetra::CreateMap<LO, GO, NO>(xpetraLib, "Cartesian3D", Comm, GaleriList); // RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout)); nodeMap->describe(*fancy,VERB_EXTREME);
            UniqueMap = Xpetra::MapFactory<LO, GO, NO>::Build(UniqueNodeMap, 3);
            Coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<SC, LO, GO, Map<LO, GO, NO>, MultiVector<SC, LO, GO, NO> >("3D", UniqueMap, GaleriList);
            RCP<Galeri::Xpetra::Problem<Map<LO, GO, NO>, CrsMatrixWrap<SC, LO, GO, NO>, MultiVector<SC, LO, GO, NO> > > Problem = Galeri::Xpetra::BuildProblem<SC, LO, GO, Map<LO, GO, NO>, CrsMatrixWrap<SC, LO, GO, NO>, MultiVector<SC, LO, GO, NO> >("Elasticity3D", UniqueMap, GaleriList);
            K = Problem->BuildMatrix();
        }


        RCP<Map<LO, GO, NO> > FullRepeatedMap;
        RCP<Map<LO, GO, NO> > RepeatedMap;
        RCP<const Map<LO, GO, NO> > FullRepeatedMapNode;
        if (useGeoMap) {
            if (Dimension == 2) {
                FullRepeatedMap = BuildRepeatedMapGaleriStruct2D<SC, LO, GO, NO>(K, M, Dimension);
                RepeatedMap = FullRepeatedMap;
            }
            else if (Dimension == 3) {
                FullRepeatedMapNode = BuildRepeatedMapGaleriStruct3D<SC, LO, GO, NO>(K->getMap(), M, Dimension);
                FullRepeatedMap = BuildMapFromNodeMap(FullRepeatedMapNode, Dimension, NodeWise);
                //FullRepeatedMapNode->describe(*fancy,Teuchos::VERB_EXTREME);
                RepeatedMap = FullRepeatedMap;
            }
        }
        else {
            RepeatedMap = BuildRepeatedMapNonConst<LO, GO, NO>(K->getCrsGraph());
        }


        RCP<MultiVector<SC, LO, GO, NO> > xSolution = MultiVectorFactory<SC, LO, GO, NO>::Build(UniqueMap, 1);
        RCP<MultiVector<SC, LO, GO, NO> > xRightHandSide = MultiVectorFactory<SC, LO, GO, NO>::Build(UniqueMap, 1);

        xSolution->putScalar(ScalarTraits<SC>::zero());
        xRightHandSide->putScalar(ScalarTraits<SC>::one());

        CrsMatrixWrap<SC, LO, GO, NO>& crsWrapK = dynamic_cast<CrsMatrixWrap<SC, LO, GO, NO>&>(*K);
        RCP<const LinearOpBase<SC> > K_thyra = ThyraUtils<SC, LO, GO, NO>::toThyra(crsWrapK.getCrsMatrix());
        RCP<MultiVectorBase<SC> >thyraX = rcp_const_cast<MultiVectorBase<SC>>(ThyraUtils<SC, LO, GO, NO>::toThyraMultiVector(xSolution));
        RCP<const MultiVectorBase<SC> >thyraB = ThyraUtils<SC, LO, GO, NO>::toThyraMultiVector(xRightHandSide);

        //-----------Set Coordinates and RepMap in ParameterList--------------------------
        RCP<ParameterList> plList = sublist(parameterList, "Preconditioner Types");
        sublist(plList, "FROSch")->set("Dimension", Dimension);
        sublist(plList, "FROSch")->set("Overlap", Overlap);
        sublist(plList, "FROSch")->set("DofOrdering", "NodeWise");
        sublist(plList, "FROSch")->set("DofsPerNode", Dimension);

        sublist(plList, "FROSch")->set("Repeated Map", RepeatedMap);
        sublist(plList, "FROSch")->set("Coordinates List", Coordinates);


        Comm->barrier();
        if (Comm->getRank() == 0) {
            cout << "##################\n# Parameter List #\n##################" << endl;
            parameterList->print(cout);
            cout << endl;
        }

        Comm->barrier(); if (Comm->getRank() == 0) cout << "###################################\n# Stratimikos LinearSolverBuilder #\n###################################\n" << endl;
        Stratimikos::LinearSolverBuilder<SC> linearSolverBuilder;
        Stratimikos::enableFROSch<SC, LO, GO, NO>(linearSolverBuilder);
        linearSolverBuilder.setParameterList(parameterList);

        Comm->barrier(); if (Comm->getRank() == 0) cout << "######################\n# Thyra PrepForSolve #\n######################\n" << endl;

        RCP<LinearOpWithSolveFactoryBase<SC> > lowsFactory =
            linearSolverBuilder.createLinearSolveStrategy("");

        lowsFactory->setOStream(out);
        lowsFactory->setVerbLevel(VERB_HIGH);


        Comm->barrier(); if (Comm->getRank() == 0) cout << "###########################\n# Thyra LinearOpWithSolve #\n###########################" << endl;

        RCP<LinearOpWithSolveBase<SC> > lows =
            linearOpWithSolve(*lowsFactory, K_thyra);

        Comm->barrier(); if (Comm->getRank() == 0) cout << "\n#########\n# Solve #\n#########" << endl;
        SolveStatus<SC> status =
            solve<SC>(*lows, Thyra::NOTRANS, *thyraB, thyraX.ptr());

        Comm->barrier(); if (Comm->getRank() == 0) cout << "\n#############\n# Finished! #\n#############" << endl;
    }


    CommWorld->barrier();
    stackedTimer->stop("Thyra Elasticity Test");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report(*out, CommWorld, options);
    std::string watchrProblemName = std::string("FROSch Elasticity ") + std::to_string(Comm->getSize()) + " ranks";
    auto xmlOut = stackedTimer->reportWatchrXML(watchrProblemName, Comm);
    if (xmlOut.length())
        std::cout << "\nAlso created Watchr performance report " << xmlOut << '\n';

    return(EXIT_SUCCESS);

}







//
//
//using Scalar = Tpetra::Details::DefaultTypes::scalar_type;
//using LO = Tpetra::Details::DefaultTypes::local_ordinal_type;
//using GO = Tpetra::Details::DefaultTypes::global_ordinal_type;
//using Node = Tpetra::Details::DefaultTypes::node_type;
//
//using Tpetra_Map = Tpetra::Map<LO, GO, Node>;
//using Tpetra_CrsMatrix = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
//using Tpetra_MultiVector = Tpetra::MultiVector<Scalar, LO, GO, Node>;
//using Tpetra_Operator = Tpetra::Operator<Scalar, LO, GO, Node>;
////using ScalarTraits = Teuchos::ScalarTraits<Scalar>;
//
//
//
//int main(int argc, char** argv) {
//
//	Tpetra::ScopeGuard scope(&argc, &argv);
//	auto comm = Tpetra::getDefaultComm();
//
//	Teuchos::ParameterList mapParameters;
//	// dimension of the problem is nx x ny
//	mapParameters.set("nx", 200 * comm->getSize());
//	mapParameters.set("ny", 200);
//	// total number of processors is mx x my
//	mapParameters.set("mx", comm->getSize());
//	mapParameters.set("my", 1);
//
//	auto map = Galeri::Xpetra::CreateMap<LO, GO, Tpetra_Map>("Cartesian2D", comm, mapParameters);
//
//	auto problem = Galeri::Xpetra::BuildProblem<
//		Scalar, LO, GO, Tpetra_Map, Tpetra_CrsMatrix, Tpetra_MultiVector>(
//			"Laplace2D", map, mapParameters);
//
//	// Build Matrix and MultiVectors
//	Teuchos::RCP<Tpetra_CrsMatrix> A = problem->BuildMatrix();
//	Teuchos::RCP<Tpetra_MultiVector>  X = Teuchos::rcp(new Tpetra_MultiVector(A->getDomainMap(), 1));
//	Teuchos::RCP<Tpetra_MultiVector>  B = Teuchos::rcp(new Tpetra_MultiVector(A->getRangeMap(), 1));
//	Teuchos::RCP<Tpetra_MultiVector>  Xexact = Teuchos::rcp(new Tpetra_MultiVector(A->getDomainMap(), 1));
//
//	Xexact->randomize(); // 참해 x를 만들고
//	X->putScalar(Teuchos::ScalarTraits<Scalar>::zero()); // 초기해 x=0
//	A->apply(*Xexact, *B); // B = A*Xexact
//
//
//	auto out = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
//
//	// ---- 2) Tpetra -> Thyra 래핑 (어댑터 사용) ----
//	auto domain = Thyra::createVectorSpace<Scalar, LO, GO, Node>(A->getDomainMap());
//	auto range = Thyra::createVectorSpace<Scalar, LO, GO, Node>(A->getRangeMap());
//
//	auto thyraA =
//		Thyra::createConstLinearOp<Scalar, LO, GO, Node>(
//			Teuchos::rcp_const_cast<const Tpetra_CrsMatrix>(A), range, domain);
//
//	auto thyraX = Thyra::createMultiVector<Scalar, LO, GO, Node>(X, range, Teuchos::null);
//	auto thyraB = Thyra::createConstMultiVector<Scalar, LO, GO, Node>(B, range, Teuchos::null);
//
//
//	//Teuchos::ParameterList::PrintOptions po;
//	//po.indent(2).showTypes(true).showDoc(true); // 문서/형식 같이 출력
//
//	//if (comm->getRank() == 0) *out << "================ Stratimikos: Valid Parameters ================\n";
//	//{
//	//	Stratimikos::DefaultLinearSolverBuilder builder;
//	//	// 등록된 프리컨까지 함께 보여주고 싶으면 활성화
//	//	Stratimikos::enableMueLu<Scalar, LO, GO, Node>(builder);
//	//	Stratimikos::enableFROSch<Scalar, LO, GO, Node>(builder);
//
//	//	Teuchos::RCP<const Teuchos::ParameterList> v = builder.getValidParameters();
//	//	if (comm->getRank() == 0 && !v.is_null()) v->print(*out, po);
//	//}
//	//{
//	//	Stratimikos::DefaultLinearSolverBuilder builder;
//	//	Stratimikos::enableMueLu<Scalar, LO, GO, Node>(builder);
//	//	Stratimikos::enableFROSch<Scalar, LO, GO, Node>(builder);
//
//	//	builder.getValidParameters()->print(std::cout, Teuchos::ParameterList::PrintOptions().showDoc(true));
//	//	for (auto s : { "KLU2","Tacho","Basker","SuperLU","SuperLUDist","MUMPS" }) {
//	//		std::cout << s << " : " << (Amesos2::query(s) ? "available" : "NO") << "\n";
//	//	}
//	//}
//
//
//
//	//if (comm->getRank() == 0) *out << "\n================ Amesos2: Available Solvers & Options ================\n";
//	//{
//	//	// 사용 가능한 직접해법만 골라 목록과 옵션을 출력
//	//	const char* names[] = {
//	//	  "KLU2","Basker","SuperLU","SuperLUDist","MUMPS","Cholmod","Umfpack","Tacho","LAPACK"
//	//	};
//	//	for (const char* s : names) {
//	//		if (!Amesos2::query(s)) continue;
//	//		if (comm->getRank() == 0) *out << "\n--- Amesos2 solver: " << s << " ---\n";
//	//		try {
//	//			// 주: getValidParameters()를 얻기 위해 실제 인스턴스가 필요
//	//			auto Ac = Teuchos::rcp_const_cast<const Tpetra::CrsMatrix<Scalar, LO, GO, Node>>(A);
//	//			auto Xnc = Teuchos::rcp_const_cast<Tpetra_MultiVector>(X);
//	//			auto Bnc = Teuchos::rcp_const_cast<Tpetra_MultiVector>(Teuchos::rcp_dynamic_cast<const Tpetra_MultiVector>(B, true));
//	//			auto solver = Amesos2::create<Tpetra::CrsMatrix<Scalar, LO, GO, Node>, Tpetra_MultiVector>(s, Ac, Xnc, Bnc);
//	//			auto v = solver->getValidParameters();
//	//			if (comm->getRank() == 0 && !v.is_null()) v->print(*out, po);
//	//		}
//	//		catch (std::exception const& e) {
//	//			if (comm->getRank() == 0) *out << "(failed to create or print: " << e.what() << ")\n";
//	//		}
//	//	}
//	//}
//
//
//	auto params = Teuchos::parameterList();
//
//
//	////---------------------------------------
//	//{
//	//	params->set("Linear Solver Type", "Belos");
//	//	params->set("Preconditioner Type", "MueLu");
//	//	{
//	//		auto& belos = params->sublist("Linear Solver Types").sublist("Belos");
//	//		belos.set("Solver Type", "Pseudo Block GMRES");
//	//		auto& gmres = belos.sublist("Solver Types").sublist("Pseudo Block GMRES");
//	//		gmres.set("Maximum Iterations", 1000);
//	//		gmres.set("Convergence Tolerance", 1e-8);
//	//		gmres.set("Output Frequency", 1);
//	//		gmres.set("Verbosity", 1 + 2 + 4 + 16 + 64); // Errors+Warnings+Iter+Final+Status
//	//	}
//	//}
//
//
//
//	{
//		auto Xcopy = Teuchos::rcp(new Tpetra_MultiVector(X->getMap(), X->getNumVectors()));
//		Tpetra::deep_copy(*Xcopy, *X);
//
//		// X, B는 RCP로 준비되어 있다고 가정
//		auto solver = Amesos2::create<Tpetra_CrsMatrix, Tpetra_MultiVector>(
//			"KLU2", A, Xcopy, B); // "KLU2" "Tacho"
//
//		//Teuchos::ParameterList a2p; // (선택) 솔버별 파라미터 넣기
//		//solver->setParameters(Teuchos::rcp(&a2p, false));
//
//		solver->describe(*out, Teuchos::EVerbosityLevel::VERB_HIGH);
//
//		Teuchos::RCP<const Teuchos::ParameterList> validParams = solver->getValidParameters();
//		validParams->print(std::cout, 2, true, true);
//
//
//		solver->symbolicFactorization();
//		solver->numericFactorization();
//		solver->solve();
//
//
//		{
//			Tpetra::MultiVector<Scalar, LO, GO, Node> R(*B); // R = B
//			Tpetra::MultiVector<Scalar, LO, GO, Node> AX(B->getMap(), B->getNumVectors());
//
//			// AX = A * X
//			A->apply(*Xcopy, AX);
//
//			// R = B - AX
//			R.update(1.0, *B, -1.0, AX, 0.0);
//
//			// 이제 R의 norm2 계산
//			std::vector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType> norms(R.getNumVectors());
//			R.norm2(norms);
//
//			if (comm->getRank() == 0) {
//				std::cout << "Residual 2-norm = " << norms[0] << std::endl;
//			}
//		}
//	}
//
//
//
//
//	//{
//
//	//	// 2) Tpetra -> Xpetra(Crs) 어댑터
//	//	Teuchos::RCP<const Xpetra::TpetraCrsMatrix<Scalar, LO, GO, Node>> xCrsA_impl =
//	//		Teuchos::rcp(new Xpetra::TpetraCrsMatrix<Scalar, LO, GO, Node>(A));
//
//	//	// 3) (업캐스트) Xpetra::CrsMatrix<Base> 로 변환
//	//	Teuchos::RCP<const Xpetra::CrsMatrix<Scalar, LO, GO, Node>> xCrsA =
//	//		Teuchos::rcp_implicit_cast<const Xpetra::CrsMatrix<Scalar, LO, GO, Node>>(xCrsA_impl);
//
//	//	// 4) FROSch 가 받는 Xpetra::Matrix 로 래핑
//	//	Teuchos::RCP<Xpetra::Matrix<Scalar, LO, GO, Node>> xA =
//	//		Teuchos::rcp(new Xpetra::CrsMatrixWrap<Scalar, LO, GO, Node>(xCrsA));
//
//	//	// ---- (2) FROSch 프리컨디셔너 생성/셋업 (Xpetra Matrix 필요) ----
//	//	Teuchos::RCP<Teuchos::ParameterList> froschParams = Teuchos::rcp(new Teuchos::ParameterList);
//	//	auto& p = *froschParams;
//	//	p.set("Overlapping", 1);
//	//	p.set("Dimension", 3);
//	//	p.set("DofsPerNode", 1);
//	//	p.set("CoarseSpace", "GDSW");
//	//	p.sublist("CoarseSolver").set("Type", "Amesos2-KLU");
//
//	//	using PrecT = FROSch::AlgebraicOverlappingPreconditioner<Scalar, LO, GO, Node>;
//	//	Teuchos::RCP<PrecT> prec = Teuchos::rcp(new PrecT(xA, froschParams));
//	//	prec->initialize();
//	//	prec->compute();
//
//	//	// ---- (3) Belos가 먹을 수 있게 Xpetra 연산자를 감싼다 ----
//	//	using BOP = Belos::OperatorT<Tpetra_MultiVector>;
//	//	Teuchos::RCP<BOP> Aop =
//	//		Teuchos::rcp(new Belos::XpetraOp<Scalar, LO, GO, Node>(xA));
//	//	Teuchos::RCP<BOP> Mop =
//	//		Teuchos::rcp(new Belos::XpetraOp<Scalar, LO, GO, Node>(prec));
//
//	//	// ---- (4) Belos 문제 설정 및 풀이 ----
//	//	Belos::LinearProblem<Scalar, Tpetra_MultiVector, BOP> problem(Aop, X, B);
//	//	problem.setRightPrec(Mop);
//	//	problem.setProblem();
//
//	//	Teuchos::ParameterList belosPL;
//	//	belosPL.set("Maximum Iterations", 200);
//	//	belosPL.set("Convergence Tolerance", 1e-8);
//
//	//	auto solver = Teuchos::rcp(
//	//		new Belos::PseudoBlockGmresSolMgr<Scalar, Tpetra_MultiVector, BOP>(Teuchos::rcpFromRef(problem),
//	//			Teuchos::rcp(&belosPL, false)));
//	//	auto ret = solver->solve();
//	//}
//
//
//
//	//---------------------------------------
//	{
//		params->set("Linear Solver Type", "Belos"); // Belos / Amesos2
//		params->set("Preconditioner Type", "Ifpack2"); // Ifpack2 / MueLu / FROSch(안됨)
//
//		auto& belos = params->sublist("Linear Solver Types").sublist("Belos");
//		belos.set("Solver Type", "Pseudo Block GMRES");
//
//		auto& ifp = params->sublist("Preconditioner Types").sublist("Ifpack2");
//
//		{
//			ifp.set("Prec Type", "RELAXATION");
//			ifp.set("Overlap", 1);
//
//			auto& s = ifp.sublist("Ifpack2 Settings");
//			s.set("relaxation: type", "Symmetric Gauss-Seidel"); // "Jacobi"도 가능
//			s.set("relaxation: sweeps", 2);
//			s.set("relaxation: damping factor", 1.0);
//		}
//
//		{
//			ifp.set("Prec Type", "ILUT");
//			ifp.set("Overlap", 1);
//
//			auto& s = ifp.sublist("Ifpack2 Settings");
//			s.set("fact: ilut level-of-fill", 1.0);
//			s.set("fact: drop tolerance", 1e-3);
//		}
//
//		{
//			ifp.set("Prec Type", "RILUK");
//
//			auto& s = ifp.sublist("Ifpack2 Settings");
//			s.set("fact: iluk level-of-fill", 2);
//			s.set("fact: iluk level-of-overlap", 0);
//		}
//
//		{
//			ifp.set("Prec Type", "CHEBYSHEV");
//
//			auto& s = ifp.sublist("Ifpack2 Settings");
//			s.set("chebyshev: degree", 3);
//			s.set("chebyshev: min eigenvalue", 1e-3);
//			s.set("chebyshev: max eigenvalue", 1.0);
//		}
//
//		{
//			ifp.set("Prec Type", "ILUT");         // 서브도메인 프리컨 타입
//			ifp.set("Overlap", 1);
//
//			auto& s = ifp.sublist("Ifpack2 Settings");
//			s.set("schwarz: overlap level", 1);
//			s.set("schwarz: use reordering", true);
//			s.set("partitioner: type", "greedy");  // 또는 Zoltan2 사용 시 아래
//
//			// ILUT 파라미터
//			s.set("fact: ilut level-of-fill", 1.0);
//			s.set("fact: drop tolerance", 1e-3);
//
//			// 서브도메인 직접해법(옵션)
//			s.set("Amesos2 solver name", "KLU2");
//		}
//
//
//
//		//auto& a2 = params->sublist("Linear Solver Types").sublist("Amesos2");
//		////a2.set("Solver Type", "KLU2");
//		//a2.set("Solver Type", "Tacho");
//		//a2.sublist("VerboseObject").set("Verbosity Level", "high");
//
//		//// Tacho 서브리스트
//		//Teuchos::ParameterList& tacho = a2.sublist("Tacho");
//		//tacho.set("method", "chol");                         // string: chol, lu 등
//		//tacho.set("variant", 2);                             // int
//		//tacho.set("small problem threshold size", 1024);     // int
//		//tacho.set("verbose", false);                         // bool
//		//tacho.set("num-streams", 1);                         // int
//		//tacho.set("dofs-per-node", 1);                       // int
//		//tacho.set("perturb-pivot", false);                   // bool
//		//tacho.set("Transpose", false);                       // bool
//
//	}
//
//
//
//	//---------------------------------------
//	{
//		params->set("Linear Solver Type", "Belos"); // Belos / Amesos2
//		params->set("Preconditioner Type", "MueLu");
//
//		auto& mue = params->sublist("Preconditioner Types").sublist("MueLu");
//		mue.set("verbosity", "high");
//		mue.set("max levels", 3);
//		mue.set("number of equations", 1);
//		mue.set("problem: symmetric", true);
//
//		mue.set("aggregation: type", "uncoupled");
//		mue.set("sa: damping factor", 1.33);
//
//		mue.set("coarse: max size", 5000);
//		mue.set("coarse: type", "Amesos2-KLU2");  // 빌드에 맞게
//
//		mue.set("smoother: type", "RELAXATION");
//		auto& sp = mue.sublist("smoother: params");
//		sp.set("relaxation: type", "Symmetric Gauss-Seidel");
//		sp.set("relaxation: sweeps", 2);
//		sp.set("relaxation: damping factor", 1.0);
//
//	}
//
//
//
//	{
//
//		// === 상위 선택 ===
//		params->set("Linear Solver Type", "Belos");
//		params->set("Preconditioner Type", "FROSch");
//
//		// === Belos 설정 ===
//		{
//			auto& lsTypes = params->sublist("Linear Solver Types");
//			auto& belos = lsTypes.sublist("Belos");
//			belos.set("Solver Type", "Block GMRES");
//
//			auto& solverTypes = belos.sublist("Solver Types");
//			auto& blockGMRES = solverTypes.sublist("Block GMRES");
//			blockGMRES.set("PreconditionerPosition", "left");
//			blockGMRES.set("Block Size", 1);
//			blockGMRES.set("Convergence Tolerance", 1e-4);
//			blockGMRES.set("Maximum Iterations", 100);
//			blockGMRES.set("Output Frequency", 1);
//			blockGMRES.set("Show Maximum Residual Norm Only", true);
//		}
//
//		// === FROSch 설정 ===
//		{
//			auto& precTypes = params->sublist("Preconditioner Types");
//			auto& frosch = precTypes.sublist("FROSch");
//
//			// 최상위 FROSch 키
//			frosch.set("FROSch Preconditioner Type", "GDSWPreconditioner");
//
//			// AlgebraicOverlappingOperator
//			{
//				auto& aoo = frosch.sublist("AlgebraicOverlappingOperator");
//				auto& solver = aoo.sublist("Solver");
//				solver.set("SolverType", "Ifpack2");
//				solver.set("Solver", "RILUK");
//
//				auto& ifp = solver.sublist("Ifpack2");
//				ifp.set("fact: iluk level-of-fill", 3);
//				// Ifpack2에서 HTS 삼각해를 강제하려면(원하실 경우):
//				// ifp.set("trisolver: type", "HTS");
//			}
//
//			// GDSWCoarseOperator
//			{
//				auto& gdsw = frosch.sublist("GDSWCoarseOperator");
//
//				// Blocks / 1
//				{
//					auto& blocks = gdsw.sublist("Blocks");
//					auto& b1 = blocks.sublist("1");
//					b1.set("Use For Coarse Space", true);
//					b1.set("Rotations", true);
//				}
//
//				// ExtensionSolver
//				{
//					auto& ext = gdsw.sublist("ExtensionSolver");
//					ext.set("SolverType", "Amesos2");
//					ext.set("Solver", "Klu");   // 보통은 "KLU2"를 많이 사용합니다(빌드에 따라 확인 권장).
//				}
//
//				// Distribution
//				{
//					auto& dist = gdsw.sublist("Distribution");
//					dist.set("Type", "linear");
//					dist.set("NumProcs", 1);
//					dist.set("Factor", 1.0);
//					dist.set("GatheringSteps", 1);
//
//					auto& comm = dist.sublist("Gathering Communication");
//					comm.set("Send type", "Send");
//				}
//
//				// CoarseSolver
//				{
//					auto& coarse = gdsw.sublist("CoarseSolver");
//					coarse.set("SolverType", "Amesos2");
//					coarse.set("Solver", "Klu"); // 필요시 "KLU2"로 교체
//				}
//			}
//		}
//	}
//
//
//
//	////---------------------------------------
//	//{
//	//	params->set("Linear Solver Type", "Amesos2"); // Belos / Amesos2
//	//	params->set("Preconditioner Type", "None");
//
//	//	auto& a2 = params->sublist("Linear Solver Types").sublist("Amesos2");
//	//	a2.set("Solver Type", "Tacho");                          // "KLU2" 등
//	//	a2.set("Refactorization Policy", "RepivotOnRefactorization");
//	//	a2.set("Throw on Preconditioner Input", true);
//	//	a2.sublist("VerboseObject").set("Verbosity Level", "high");
//
//	//	// 솔버별 서브리스트를 같은 곳에 추가 (Amesos2 내부로 그대로 전달됨)
//	//	auto& tacho = a2.sublist("Tacho");
//	//	tacho.set("method", "chol");
//	//	tacho.set("variant", 2);
//	//	tacho.set("small problem threshold size", 1024);
//	//	tacho.set("verbose", false);
//	//	tacho.set("num-streams", 1);
//	//	tacho.set("dofs-per-node", 1);
//	//	tacho.set("perturb-pivot", false);
//	//	tacho.set("Transpose", false);
//	//}
//
//
//	//params->validateParametersAndSetDefaults(*params);
//
//
//	{
//		Stratimikos::DefaultLinearSolverBuilder builder;
//		Stratimikos::enableMueLu<Scalar, LO, GO, Node>(builder);
//		Stratimikos::enableFROSch<Scalar, LO, GO, Node>(builder);
//
//		// ---- 4) 파라미터 설정 (Belos + MueLu) ----
//		builder.setParameterList(params);
//
//		params->print(std::cout);
//
//
//		// ---- 5) Solver 생성/해 풀기 ----
//		X->putScalar(Teuchos::ScalarTraits<Scalar>::zero()); // 초기해 리셋
//		auto solverFactory = Thyra::createLinearSolveStrategy(builder);
//		auto lows = Thyra::linearOpWithSolve(*solverFactory, thyraA);
//
//		auto status = Thyra::solve<Scalar>(*lows, Thyra::NOTRANS, *thyraB, thyraX.ptr());
//
//		if (comm->getRank() == 0) {
//			std::cout << "Solve status: " << Thyra::toString(status.solveStatus)
//				<< ", achieved tol = " << status.achievedTol << "\n";
//		}
//
//	}
//
//
//
//
//	//// =====================================================================================
//	//// 4) Belos + FROSch  (좌표/Repeated Map 제공)
//	//// =====================================================================================
//	//{
//	//	Stratimikos::DefaultLinearSolverBuilder builder;
//	//	Stratimikos::enableFROSch<Scalar, LO, GO, Node>(builder); // Stratimikos에 FROSch 등록
//
//	//	// 좌표(2D)와 Repeated Map 생성
//	//	//auto coords = Galeri::Xpetra::CreateCartesianCoordinates<
//	//	//	Scalar, LO, GO, Tpetra_Map, Tpetra_MultiVector>("2D", A->getRowMap(), mapParameters);
//	//	auto repMap = A->getColMap();
//
//	//	//-----------Set Coordinates and RepMap in ParameterList--------------------------
//
//	//	auto params = Teuchos::parameterList();
//	//	params->set("Linear Solver Type", "Belos");
//	//	params->set("Preconditioner Type", "FROSch");
//
//	//	//// Belos: GMRES
//	//	//{
//	//	//	auto& belos = params->sublist("Linear Solver Types").sublist("Belos");
//	//	//	belos.set("Solver Type", "Pseudo Block GMRES");
//	//	//	auto& gmres = belos.sublist("Solver Types").sublist("Pseudo Block GMRES");
//	//	//	gmres.set("Maximum Iterations", 1000);
//	//	//	gmres.set("Convergence Tolerance", 1e-8);
//	//	//	gmres.set("Output Frequency", 1);
//	//	//	gmres.set("Verbosity", 1 + 2 + 4 + 16 + 64);
//	//	//}
//
//
//	//	//// FROSch 설정 (단일 블록 예제)
//	//	//{
//	//	//	auto& precTypes = params->sublist("Preconditioner Types");
//	//	//	auto& frosch = precTypes.sublist("FROSch");
//	//	//	frosch.set("Dimension", 2);
//	//	//	frosch.set("Overlap", 1);
//	//	//	//frosch.set("Coordinates", coords); // RCP<const Tpetra_MultiVector>
//	//	//	frosch.set("Repeated Map", repMap); // RCP<const Tpetra_Map>
//	//	//	frosch.set("DofOrdering", "NodeWise");
//	//	//	frosch.set("DofsPerNode", 1);
//	//	//	// 필요 시 추가 파라미터: frosch.sublist("Partitioner") ...
//	//	//}
//
//	//	builder.setParameterList(params);
//
//	//	// 풀기
//	//	X->putScalar(Teuchos::ScalarTraits<Scalar>::zero()); // 초기해 리셋
//	//	auto solverFactory = Thyra::createLinearSolveStrategy(builder);
//	//	solverFactory->setOStream(out);
//	//	solverFactory->setVerbLevel(Teuchos::VERB_HIGH);
//
//	//	auto lows = Thyra::linearOpWithSolve(*solverFactory, thyraA);
//	//	auto status = Thyra::solve<Scalar>(*lows, Thyra::NOTRANS, *thyraB, thyraX.ptr());
//
//	//	if (comm->getRank() == 0) {
//	//		std::cout << "Solve status (FROSch): " << Thyra::toString(status.solveStatus)
//	//			<< ", achieved tol = " << status.achievedTol << "\n";
//	//	}
//	//}
//
//}
//
//
//
//
////
////// ===== 결과 래퍼 =====
////template<class Scalar>
////struct TimedResult {
////    bool ok{ false };
////    double residual{ std::numeric_limits<double>::quiet_NaN() };
////    int iters{ -1 };      // 직접해법은 -1, Belos 반복수 파싱은 생략(간략화)
////    double milliseconds{ 0.0 };
////    std::string what;
////};
////
////// ===== 2D Poisson(Dirichlet 0) 수동 조립 =====
////template<class Scalar, class LO, class GO, class Node>
////Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LO, GO, Node>>
////build_poisson_2d_manual(
////    int N,
////    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& map)
////{
////    using crs_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
////    auto A = Teuchos::rcp(new crs_type(map, 5));
////    const size_t nLocal = map->getLocalNumElements();
////
////    for (size_t lrow = 0; lrow < nLocal; ++lrow) {
////        const GO gid = map->getGlobalElement(static_cast<LO>(lrow));
////        const int i = static_cast<int>(gid % N);
////        const int j = static_cast<int>(gid / N);
////        const bool on_bnd = (i == 0 || i == N - 1 || j == 0 || j == N - 1);
////
////        std::vector<GO> idx;
////        std::vector<Scalar> val;
////
////        if (on_bnd) {
////            idx.push_back(gid); val.push_back(Scalar(1));  // u=0 Dirichlet
////        }
////        else {
////            idx.reserve(5); val.reserve(5);
////            idx.push_back(gid - 1); val.push_back(Scalar(-1));
////            idx.push_back(gid + 1); val.push_back(Scalar(-1));
////            idx.push_back(gid - N); val.push_back(Scalar(-1));
////            idx.push_back(gid + N); val.push_back(Scalar(-1));
////            idx.push_back(gid);     val.push_back(Scalar(4));
////        }
////
////        Teuchos::ArrayView<const GO> cols(idx.data(), (int)idx.size());
////        Teuchos::ArrayView<const Scalar> vals(val.data(), (int)val.size());
////        A->insertGlobalValues(gid, cols, vals);
////    }
////    A->fillComplete();
////    return A;
////}
////
////#ifdef USE_GALERI
////// ===== Galeri(Xpetra)로 Laplace2D 생성 =====
////template<class Scalar, class LO, class GO, class Node>
////Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LO, GO, Node>>
////build_poisson_2d_galeri(
////    int N,
////    const Teuchos::RCP<const Tpetra::Map<LO, GO, Node>>& tpetraMap)
////{
////    using TpetCrs = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
////    using XTMap = Xpetra::Map<LO, GO, Node>;
////    using XTCrsW = Xpetra::TpetraCrsMatrix<Scalar, LO, GO, Node>;
////    using XTMapTpl = Xpetra::TpetraMap<LO, GO, Node>;
////    Teuchos::RCP<const XTMap> xMap = Teuchos::rcp(new XTMapTpl(tpetraMap));
////
////    Teuchos::ParameterList galeriParams;
////    galeriParams.set("nx", N);
////    galeriParams.set("ny", N);
////    // 분할 파라미터 (필요 시)
////    galeriParams.set("mx", tpetraMap->getComm()->getSize());
////    galeriParams.set("my", 1);
////
////    auto problem =
////        Galeri::Xpetra::BuildProblem<Scalar, LO, GO, XTMap, XTCrsW>(
////            "Laplace2D", xMap, galeriParams);
////
////    auto Awrap = problem->BuildMatrix();
////    return Teuchos::rcp_dynamic_cast<XTCrsW>(Awrap)->getTpetra_CrsMatrixNonConst();
////
////}
////#endif
////
////// ===== ||B - A*X||_2 =====
////template<class Scalar, class LO, class GO, class Node>
////typename Teuchos::ScalarTraits<Scalar>::magnitudeType
////residual_norm2(const Tpetra::Operator<Scalar, LO, GO, Node>& Aop,
////    const Tpetra::Vector<Scalar, LO, GO, Node>& X,
////    const Tpetra::Vector<Scalar, LO, GO, Node>& B)
////{
////    using mv_type = Tpetra::MultiVector<Scalar, LO, GO, Node>;
////    mv_type R(B.getMap(), 1);
////    Aop.apply(X, R);                         // R = A*X
////    R.update(Scalar(1), B, Scalar(-1));      // R = B - A*X
////    std::vector<typename Teuchos::ScalarTraits<Scalar>::magnitudeType> nrm(1);
////    R.norm2(nrm);
////    return nrm[0];
////}
////
////// ===== Amesos2 직접해법 실행 =====
////template<class Scalar, class LO, class GO, class Node>
////TimedResult<Scalar>
////run_amesos2(const std::string& name,
////    const Teuchos::RCP<const Tpetra::CrsMatrix<Scalar, LO, GO, Node>>& A,
////    Tpetra::Vector<Scalar, LO, GO, Node>& X,
////    const Tpetra::Vector<Scalar, LO, GO, Node>& B,
////    const Teuchos::ParameterList* opt = nullptr)
////{
////    using crs_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
////    using mv_type = Tpetra::MultiVector<Scalar, LO, GO, Node>;
////    TimedResult<Scalar> out;
////
////    try {
////        auto Aconst = Teuchos::rcp_const_cast<const crs_type>(A);
////        auto Xrcp = Teuchos::rcp(&X, false);
////        auto Brcp = Teuchos::rcp(const_cast<Tpetra::Vector<Scalar, LO, GO, Node>*>(&B), false);
////
////        auto solver = Amesos2::create<crs_type, mv_type>(name, Aconst, Xrcp, Brcp);
////        if (opt) solver->setParameters(Teuchos::rcp(const_cast<Teuchos::ParameterList*>(opt), false));
////
////        auto t0 = std::chrono::high_resolution_clock::now();
////        solver->symbolicFactorization();
////        solver->numericFactorization();
////        solver->solve();
////        auto t1 = std::chrono::high_resolution_clock::now();
////
////        out.ok = true;
////        out.residual = residual_norm2<Scalar, LO, GO, Node>(*A, X, B);
////        out.milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count();
////    }
////    catch (const std::exception& ex) {
////        out.ok = false; out.what = ex.what();
////    }
////    return out;
////}
////
////// ===== Stratimikos 반복해법 실행 =====
////template<class Scalar, class LO, class GO, class Node>
////TimedResult<Scalar>
////run_stratimikos(const Teuchos::ParameterList& plist,
////    const Teuchos::RCP<const Tpetra::CrsMatrix<Scalar, LO, GO, Node>>& A,
////    Tpetra::Vector<Scalar, LO, GO, Node>& X,
////    const Tpetra::Vector<Scalar, LO, GO, Node>& B)
////{
////    using ST = Scalar;
////    using vec_type = Tpetra::Vector<Scalar, LO, GO, Node>;
////    using Op = Tpetra::Operator<ST, LO, GO, Node>;
////    TimedResult<Scalar> out;
////    try {
////        Teuchos::RCP<const Op> A_op =
////            Teuchos::rcp_dynamic_cast<const Op>(A, /*throw_on_fail=*/true);
////        //auto A_thyra = Thyra::createLinearOp<ST, LO, GO, Node>(A_op);
////        auto A_thyra = Thyra::createLinearOp<ST, LO, GO, Node>(A);
////
////
////        auto X_mv = Teuchos::rcp(&X, false);
////        auto B_mv = Teuchos::rcp(const_cast<Tpetra::Vector<Scalar, LO, GO, Node>*>(&B), false);
////        // 비소유 RCP (수명은 호출자 보장)
////        auto X_thyra = Thyra::createVector<ST, LO, GO, Node>(Teuchos::rcp(&X, false));
////        auto B_thyra = Thyra::createConstVector<ST, LO, GO, Node>(
////            Teuchos::rcp(const_cast<vec_type*>(&B), false));
////
////        Stratimikos::DefaultLinearSolverBuilder builder;
////        builder.setParameterList(Teuchos::rcp(new Teuchos::ParameterList(plist)));
////        auto lowsFactory = builder.createLinearSolveStrategy("");
////        auto lows = Thyra::linearOpWithSolve<ST>(*lowsFactory, A_thyra);
////
////        auto t0 = std::chrono::high_resolution_clock::now();
////        auto status = Thyra::solve<ST>(*lows, Thyra::NOTRANS, *B_thyra, X_thyra.ptr());  // 벡터 오버로드
////        auto t1 = std::chrono::high_resolution_clock::now();
////
////        out.ok = (status.solveStatus == Thyra::SOLVE_STATUS_CONVERGED);
////        out.residual = residual_norm2<Scalar, LO, GO, Node>(*A, X, B);
////        out.milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count();
////        out.what = status.message;
////    }
////    catch (const std::exception& ex) {
////        out.ok = false; out.what = ex.what();
////    }
////    return out;
////}
////
////int main(int argc, char** argv)
////{
////
////
////
////
////    Tpetra::ScopeGuard guard(&argc, &argv);
////    {
////        using Scalar = Tpetra::Details::DefaultTypes::scalar_type;
////        using LO = Tpetra::Details::DefaultTypes::local_ordinal_type;
////        using GO = Tpetra::Details::DefaultTypes::global_ordinal_type;
////        using Node = Tpetra::Details::DefaultTypes::node_type;
////
////        using map_type = Tpetra::Map<LO, GO, Node>;
////        using crs_type = Tpetra::CrsMatrix<Scalar, LO, GO, Node>;
////        using vec_type = Tpetra::Vector<Scalar, LO, GO, Node>;
////
////        auto comm = Tpetra::getDefaultComm();
////        const int rank = comm->getRank();
////
////        int N = 100;
////        for (int k = 1; k + 1 <= argc; ++k)
////            if (std::string(argv[k]) == "--n" && k + 1 < argc) N = std::stoi(argv[++k]);
////
////        const GO nGlobal = static_cast<GO>(N) * static_cast<GO>(N);
////        auto map = Teuchos::rcp(new map_type(nGlobal, 0, comm));
////
////        // === 시스템(A,X,B)
////        Teuchos::RCP<crs_type> A;
////#ifdef USE_GALERI
////        A = build_poisson_2d_galeri<Scalar, LO, GO, Node>(N, map);
////#else
////        A = build_poisson_2d_manual<Scalar, LO, GO, Node>(N, map);
////#endif
////
////        auto X = Teuchos::rcp(new vec_type(map, true));
////        auto B = Teuchos::rcp(new vec_type(map, true)); B->putScalar(Scalar(1));
////        // Dirichlet 경계에서 RHS=0
////        {
////            const size_t nLocal = map->getLocalNumElements();
////            for (size_t l = 0; l < nLocal; ++l) {
////                const GO gid = map->getGlobalElement((LO)l);
////                const int i = (int)(gid % N), j = (int)(gid / N);
////                if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
////                    B->replaceLocalValue((LO)l, Scalar(0));
////            }
////        }
////
////        TimedResult<Scalar> out;
////
////        // 1) A 래핑: CrsMatrix → (const) Operator → Thyra
////        using op_type = Tpetra::Operator<Scalar, LO, GO, Node>;
////        auto A_thyra = Thyra::createLinearOp<Scalar, LO, GO, Node>(A);
////
////        // 2) 벡터 오버로드(간단/안전)
////        auto X_thyra = Thyra::createVector<Scalar, LO, GO, Node>(X);
////        auto B_thyra = Thyra::createConstVector<Scalar, LO, GO, Node>(B);
////
////
////
////
////        Teuchos::ParameterList p;
////        //auto& S = p.sublist("Stratimikos");
////        //S.set("Linear Solver Type", "Belos");
////        //S.set("Preconditioner Type", "Ifpack2");
////
////        //auto& Bp = p.sublist("Belos");
////        //Bp.set("Solver Type", "GMRES");
////        //auto& Bgm = Bp.sublist("Solver Types").sublist("GMRES");
////        //Bgm.set("Convergence Tolerance", 1e-8);
////        //Bgm.set("Maximum Iterations", 500);
////        //Bgm.set("Verbosity", 33);
////
////        //auto& Ip = p.sublist("Ifpack2");
////        //Ip.set("Preconditioner Type", "ILUT");
////        //auto& Ilut = Ip.sublist("IFPACK2").sublist("ILUT");
////        //Ilut.set("fact: ilut level-of-fill", 1.0);
////        //Ilut.set("fact: drop tolerance", 1e-3);
////
////        // Stratimikos 빌더/LOWS 생성 그대로 유지
////        Stratimikos::DefaultLinearSolverBuilder builder;
////
////
////        builder.getValidParameters()->print(std::cout, Teuchos::ParameterList::PrintOptions().showDoc(true));
////        for (auto s : { "KLU2","Tacho","Basker","SuperLU","SuperLUDist","MUMPS" }) {
////            std::cout << s << " : " << (Amesos2::query(s) ? "available" : "NO") << "\n";
////        }
////
////
////        Teuchos::ParameterList muelu;
////        muelu.set("verbosity", "none");
////        muelu.set("max levels", 3);
////        muelu.set("coarse: max size", 5000);
////        muelu.set("smoother: type", "RELAXATION");
////        muelu.sublist("smoother: params").set("relaxation: type", "Jacobi");
////        // 필요 시 aggregation, coarse solver 등 튜닝
////        MueLu::CreateTpetraPreconditioner<Scalar, LO, GO, Node>(A, muelu);
////
////
////        //// 1) Tpetra -> Xpetra 래핑
////        //using XT_Crs = Xpetra::TpetraCrsMatrix<Scalar, LO, GO, Node>;
////        //Teuchos::RCP<const Xpetra::Matrix<Scalar, LO, GO, Node>> A_xpetra =
////        //    Teuchos::rcp(new XT_Crs(A_tpetra));
////
////        //// 2) FROSch 파라미터
////        //Teuchos::RCP<Teuchos::ParameterList> prm = Teuchos::rcp(new Teuchos::ParameterList);
////        //// 예: prm->set("Overlap", 1); prm->set("Dimension", 3);
////        ////     prm->set("Algebraic Overlap", 1); 등 (문서/예제 참고)
////
////        //// 3) 전처리기 생성/초기화
////        //auto frosch = Teuchos::rcp(new FROSch::AlgebraicOverlappingPreconditioner<Scalar, LO, GO, Node>(A_xpetra, prm));
////        //frosch->initialize();
////        //frosch->compute();
////
////
////
////#ifdef HAVE_MUELU_TPETRA      // 또는 MueLu_HAVE_TPETRA (버전에 따라 다를 수 있음)
////        std::cout << "[compile] MueLu Tpetra adapter: ON\n";
////#else
////        std::cout << "[compile] MueLu Tpetra adapter: OFF\n";
////#endif
////
////
////
////        builder.setParameterList(Teuchos::rcp(new Teuchos::ParameterList(p)));
////        auto lowsFactory = builder.createLinearSolveStrategy("");
////        auto lows = Thyra::linearOpWithSolve<Scalar>(*lowsFactory, A_thyra);
////
////
////
////
////        // 풀기
////        //auto t0 = std::chrono::high_resolution_clock::now();
////        auto status = Thyra::solve<Scalar>(*lows, Thyra::NOTRANS, *B_thyra, X_thyra.ptr());
////        //auto t1 = std::chrono::high_resolution_clock::now();
////
////        
////        std::cout << status.message << std::endl;
////        std::cout << status.achievedTol << std::endl;
////
////        //out.ok = (status.solveStatus == Thyra::SOLVE_STATUS_CONVERGED);
////        //out.milliseconds = std::chrono::duration<double, std::milli>(t1 - t0).count();
////        //out.what = status.message;
////
////        //// (선택) 잔차 계산
////        //vec_type R(map, true);
////        //A->apply(*X, R);
////        //R.update(Scalar(1), *B, Scalar(-1));
////        //using mag = typename Teuchos::ScalarTraits<Scalar>::magnitudeType;
////        //std::vector<mag> nrm(1); R.norm2(nrm);
////        //out.residual = nrm[0];
////
////
////
////        //if (rank == 0) {
////        //    std::cout << "[N=" << N << "] unknowns=" << (size_t)nGlobal << "\n";
////        //    std::cout << "[init] ||r||2 = " << residual_norm2<Scalar, LO, GO, Node>(*A, *X, *B) << "\n\n";
////        //}
////
////        //// === (1) Amesos2 직접해법
////        ////std::vector<std::string> direct = { "KLU2","Basker","SuperLU","SuperLU_DIST","STRUMPACK","LAPACK" };
////        //std::vector<std::string> direct = { "KLU2","Basker","SuperLU","SuperLU_DIST","STRUMPACK" };
////        //if (rank == 0) {
////        //    std::cout << "=== Amesos2 (Direct) ===\n";
////        //    std::cout << std::left << std::setw(16) << "Solver"
////        //        << std::setw(10) << "Status"
////        //        << std::setw(18) << "||r||2"
////        //        << "time(ms)\n";
////        //    std::cout << std::string(60, '-') << "\n";
////        //}
////        //for (auto& name : direct) {
////        //    X->putScalar(Scalar(0));
////        //    auto ret = run_amesos2<Scalar, LO, GO, Node>(name, A, *X, *B, nullptr);
////        //    if (rank == 0) {
////        //        std::cout << std::left << std::setw(16) << name
////        //            << std::setw(10) << (ret.ok ? "OK" : "FAIL")
////        //            << std::setw(18) << (ret.ok ? std::to_string(ret.residual) : ret.what.substr(0, 16))
////        //            << ret.milliseconds << "\n";
////        //    }
////        //}
////        //if (rank == 0) std::cout << "\n";
////
////        //// === Stratimikos 공통(반복해법: Belos)
////        //auto make_ifpack2 = []() {
////        //    Teuchos::ParameterList p;
////        //    auto& S = p.sublist("Stratimikos");
////        //    S.set("Linear Solver Type", "Belos");
////        //    S.set("Preconditioner Type", "Ifpack2");
////
////        //    auto& Bp = p.sublist("Belos");
////        //    Bp.set("Solver Type", "GMRES");
////        //    auto& Bgm = Bp.sublist("Solver Types").sublist("GMRES");
////        //    Bgm.set("Convergence Tolerance", 1e-8);
////        //    Bgm.set("Maximum Iterations", 500);
////        //    Bgm.set("Verbosity", 33);
////
////        //    auto& Ip = p.sublist("Ifpack2");
////        //    Ip.set("Preconditioner Type", "ILUT");
////        //    auto& Ilut = Ip.sublist("IFPACK2").sublist("ILUT");
////        //    Ilut.set("fact: ilut level-of-fill", 1.0);
////        //    Ilut.set("fact: drop tolerance", 1e-3);
////        //    return p;
////        //    };
////
////        //auto make_muelu = []() {
////        //    Teuchos::ParameterList p;
////        //    auto& S = p.sublist("Stratimikos");
////        //    S.set("Linear Solver Type", "Belos");
////        //    S.set("Preconditioner Type", "MueLu");
////
////        //    auto& Bp = p.sublist("Belos");
////        //    Bp.set("Solver Type", "CG"); // SPD
////        //    auto& Bcg = Bp.sublist("Solver Types").sublist("CG");
////        //    Bcg.set("Convergence Tolerance", 1e-8);
////        //    Bcg.set("Maximum Iterations", 500);
////        //    Bcg.set("Verbosity", 33);
////
////        //    p.sublist("MueLu").set("verbosity", "low");
////        //    return p;
////        //    };
////
////        //// ShyLU-NodeTacho를 Preconditioner로 사용 (Stratimikos에서 등록되어 있을 때)
////        //auto make_shylu_tacho = []() {
////        //    Teuchos::ParameterList p;
////        //    auto& S = p.sublist("Stratimikos");
////        //    S.set("Linear Solver Type", "Belos");
////        //    S.set("Preconditioner Type", "ShyLU");  // 노드 계열 프리컨
////        //    auto& Bp = p.sublist("Belos");
////        //    Bp.set("Solver Type", "GMRES");
////        //    auto& Bgm = Bp.sublist("Solver Types").sublist("GMRES");
////        //    Bgm.set("Convergence Tolerance", 1e-8);
////        //    Bgm.set("Maximum Iterations", 500);
////        //    Bgm.set("Verbosity", 33);
////
////        //    // ShyLU 노드 타입: "Tacho" 지정
////        //    auto& Sh = p.sublist("ShyLU");
////        //    Sh.set("Preconditioner Type", "Tacho");
////        //    // 추가 파라미터가 필요한 경우 Sh.sublist("Tacho")에 설정
////        //    return p;
////        //    };
////
////        //// FROSch (도메인 분할) — 빌드 환경에 따라 미등록일 수 있음(그 경우 FAIL로 잡힘)
////        //auto make_frosch = []() {
////        //    Teuchos::ParameterList p;
////        //    auto& S = p.sublist("Stratimikos");
////        //    S.set("Linear Solver Type", "Belos");
////        //    S.set("Preconditioner Type", "FROSch");
////        //    auto& Bp = p.sublist("Belos");
////        //    Bp.set("Solver Type", "GMRES");
////        //    auto& Bgm = Bp.sublist("Solver Types").sublist("GMRES");
////        //    Bgm.set("Convergence Tolerance", 1e-8);
////        //    Bgm.set("Maximum Iterations", 500);
////        //    Bgm.set("Verbosity", 33);
////
////        //    auto& F = p.sublist("FROSch");
////        //    // 예시 파라미터(실사용 시 문제/격자/분할에 맞게 조정)
////        //    F.set("CoarseSolverType", "Amesos2-KLU2"); // 코어스 레벨 직접해법
////        //    // F.sublist("Partitioner").set("Type","ParMETIS"); 등
////        //    return p;
////        //    };
////
////        //struct Item { const char* name; Teuchos::ParameterList(*make)(); };
////        //std::vector<Item> iter_cfg = {
////        //  {"Belos+Ifpack2",  make_ifpack2},
////        //  {"Belos+MueLu",    make_muelu},
////        //  {"Belos+ShyLU-Tacho", make_shylu_tacho},
////        //  {"Belos+FROSch",   make_frosch}
////        //};
////
////        //if (rank == 0) {
////        //    std::cout << "=== Stratimikos (Iterative) ===\n";
////        //    std::cout << std::left << std::setw(20) << "Config"
////        //        << std::setw(10) << "Status"
////        //        << std::setw(18) << "||r||2"
////        //        << "time(ms)\n";
////        //    std::cout << std::string(64, '-') << "\n";
////        //}
////        //for (auto& it : iter_cfg) {
////        //    X->putScalar(Scalar(0));
////        //    auto plist = it.make();
////        //    auto ret = run_stratimikos<Scalar, LO, GO, Node>(plist, A, *X, *B);
////        //    if (rank == 0) {
////        //        std::cout << std::left << std::setw(20) << it.name
////        //            << std::setw(10) << (ret.ok ? "OK" : "FAIL")
////        //            << std::setw(18) << (ret.ok ? std::to_string(ret.residual) : ret.what.substr(0, 16))
////        //            << ret.milliseconds << "\n";
////        //    }
////        //}
////    }
////    return 0;
////}
////
