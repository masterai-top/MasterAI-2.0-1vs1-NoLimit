#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include "mc/simulator.h"

using namespace std;

PYBIND11_MODULE(mc, m)
{
    //Monte Carlo
    pybind11::class_<Simulator>(m, "Simulator")
    .def(pybind11::init<std::string &>())
    .def("compute_probabilities", &Simulator::compute_probabilities,
         pybind11::arg("test_num"),
         pybind11::arg("comm_hand"),
         pybind11::arg("known_hands"),
         pybind11::arg("players_unknown"));
}
