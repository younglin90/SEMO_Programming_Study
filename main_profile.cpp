//// Copyright (c) 2015-2019 Pawe©© Cichocki
//// License: https://opensource.org/licenses/MIT
//
//#include <math.h>
//#include <future>
//#include <iostream>
//
//#include "iprof.hpp"
//
//using namespace std;
//
//// Senseless calculations func 1
//double bigWave()
//{
//    IPROF_FUNC;
//
//    double ret = 0;
//    for (int i = 0; i < 10000; ++i)
//        ret += sin(i / 1000) - ret * 0.9;
//    return ret;
//}
//
//// Senseless calculations func 2
//double hugePower()
//{
//    IPROF_FUNC;
//
//    double ret = 2;
//    {
//        IPROF("FirstPowerLoop");
//        for (int i = 0; i < 5000; ++i)
//        {
//            double exp = (i % 10 + 1) / 7.8;
//            ret = pow(ret * 1.4, exp);
//        }
//    }
//    {
//        IPROF("SecondPowerLoop");
//        for (int i = 0; i < 5000; ++i)
//        {
//            double exp = double(i & 15) * 0.08;
//            ret = pow(ret * 1.4, exp);
//        }
//    }
//    {
//        IPROF("BigWavePowerLoop");
//        for (int i = 0; i < 3; ++i)
//            ret -= bigWave();
//    }
//
//    return ret;
//}
//
//// Senseless calculations func 3
//double heavyCalc()
//{
//    IPROF_FUNC;
//
//    double ret = 0;
//    for (int i = 0; i < 1000; ++i)
//    {
//        ret += i;
//    }
//    return ret;
//}
//
//
//int main()
//{
//    auto startTime = HighResClock::now();
//
//    cout << "Hi ;)\n" << endl;
//
//    cout << "sizeof(InternalProfiler::Tree): " << sizeof(InternalProfiler::Tree)
//        << " bytes" << endl;
//
//    cout << "\nAnd the lucky double is: " << heavyCalc() << endl;
//    cout << "\nAnd the lucky double is: " << heavyCalc() << endl;
//    cout << "\nAnd the lucky double is: " << heavyCalc() << endl;
//
//    InternalProfiler::aggregateEntries();
//    cout << "\nThe profiler stats after the second run:\n"
//        << InternalProfiler::stats << endl;
//
//
//    cout << "The test took " << MILLI_SECS(HighResClock::now() - startTime)
//        << " milliseconds\nGoodbye" << endl;
//    return 0;
//}

//
//#include <ctrack.hpp>
//
//void a() {
//    CTRACK;
//    // Simulating some work
//    for (int i = 0; i < 1000000; ++i) {
//        // Do something
//    }
//}
//void b() {
//    CTRACK;
//    // Simulating some work
//    for (int i = 0; i < 1000000; ++i) {
//        // Do something
//    }
//}
//
//int main() {
//    for (int i = 0; i < 100; ++i) {
//        a();
//    }
//    b();
//
//    // Print results to console
//    ctrack::clear_a_store();
//    ctrack::result_print();
//
//    return 0;
//}

#include <Eigen/Core>
#include <iostream>
#include <boost/numeric/interval.hpp>
namespace interval_options {
    typedef boost::numeric::interval_lib::checking_catch_nan<double>
        CheckingPolicy;
} // namespace interval_options


int main() {


    typedef boost::numeric::interval<
        double,
        boost::numeric::interval_lib::policies<
        boost::numeric::interval_lib::save_state<
        boost::numeric::interval_lib::rounded_transc_std<double>>,
        interval_options::CheckingPolicy>>
        Interval;

    Interval ti(0, 1);

    Eigen::Vector3d B_eb0_ti0;
    Eigen::Vector3d B_eb0_ti1;

    B_eb0_ti0 << 1, 2, 3;
    B_eb0_ti1 << 4, 5, 6;

    Eigen::Vector3<Interval> eb0_ti0 = B_eb0_ti0.cast<Interval>();
    Eigen::Vector3<Interval> eb0_ti1 = B_eb0_ti1.cast<Interval>();
    Eigen::Vector3<Interval> eb0 = (B_eb0_ti1 - B_eb0_ti0).cast<Interval>() * ti + B_eb0_ti0.cast<Interval>();
    Interval d = (eb0 - ((eb0_ti1 - eb0_ti0) * ti + eb0_ti0)).norm();

    std::cout << d.lower() << std::endl;
    std::cout << d.upper() << std::endl;

    std::cout << (B_eb0_ti0 - (B_eb0_ti0)).norm() << std::endl;
    std::cout << (B_eb0_ti1 - (B_eb0_ti0)).norm() << std::endl;
    std::cout << (B_eb0_ti0 - (B_eb0_ti1)).norm() << std::endl;
    std::cout << (B_eb0_ti1 - (B_eb0_ti1)).norm() << std::endl;

}