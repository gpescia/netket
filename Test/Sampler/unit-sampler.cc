// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>
#include "catch.hpp"
#include "netket.hpp"

#include "sampler_input_tests.hpp"

TEST_CASE("sampler generates states in Hilbert space", "[sampler]") {
  auto input_tests = GetSamplerInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i]["Sampler"].dump();

    SECTION("Sampler test (" + std::to_string(i) + ") on " + name) {
      auto pars = input_tests[i];

      netket::Graph graph(pars);

      netket::Hamiltonian hamiltonian(graph, pars);

      auto &hilbert = hamiltonian.GetHilbert();
      auto localstates = hilbert.LocalStates();

      using MachineType = netket::Machine<std::complex<double>>;
      MachineType machine(graph, hamiltonian, pars);

      netket::Sampler<MachineType> sampler(graph, hamiltonian, machine, pars);

      sampler.Reset(true);

      int nsweeps = 100;

      for (int sw = 0; sw < nsweeps; sw++) {
        auto visible = sampler.Visible();

        REQUIRE(visible.size() == hilbert.Size());

        for (int k = 0; k < visible.size(); k++) {
          REQUIRE(std::count(localstates.begin(), localstates.end(),
                             visible(k)) > 0);
        }

        sampler.Sweep();
      }

      REQUIRE(sampler.Acceptance().minCoeff() > 0);
      REQUIRE(sampler.Acceptance().maxCoeff() <= 1);
    }
  }
}

// Testing that samples generated from direct sampling are compatible with those
// generated by markov chain sampling
// here we use the L_1 test presented in https://arxiv.org/pdf/1308.3946.pdf
TEST_CASE("sampler generates states correctly distributed", "[sampler]") {
  auto input_tests = GetSamplerInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i]["Sampler"].dump();

    SECTION("Sampler test (" + std::to_string(i) + ")  on " + name) {
      auto pars = input_tests[i];

      netket::Graph graph(pars);

      REQUIRE(graph.Nsites() > 0);

      netket::Hamiltonian hamiltonian(graph, pars);

      auto &hilbert = hamiltonian.GetHilbert();
      auto localstates = hilbert.LocalStates();

      netket::HilbertIndex hilb_index(hilbert);

      using MachineType = netket::Machine<std::complex<double>>;
      MachineType machine(graph, hamiltonian, pars);

      netket::Sampler<MachineType> sampler(graph, hamiltonian, machine, pars);

      sampler.Reset();

      std::vector<double> countsSampler(hilb_index.NStates(), 0);

      std::size_t nsweeps =
          std::max(10 * hilb_index.NStates(), std::size_t(10000));

      for (std::size_t sw = 0; sw < nsweeps; sw++) {
        auto visible = sampler.Visible();

        auto staten = hilb_index.StateToNumber(visible);
        countsSampler[staten] += 1.;

        sampler.Sweep();
      }

      // Now generate samples from direct sampling
      // First let's make sure that the wave-function is computed without
      // overflowing
      double logmax = -std::numeric_limits<double>::infinity();

      std::vector<MachineType::StateType> logpsi(hilb_index.NStates());
      for (std::size_t st = 0; st < hilb_index.NStates(); st++) {
        auto state = hilb_index.NumberToState(st);
        logpsi[st] = machine.LogVal(state);
        logmax = std::max(std::real(logpsi[st]), logmax);
      }

      std::vector<double> psisquare(hilb_index.NStates());

      double l2norm = 0;
      double norm = 0;
      for (std::size_t st = 0; st < hilb_index.NStates(); st++) {
        psisquare[st] = std::norm(std::exp(logpsi[st] - logmax));
        l2norm += std::pow(psisquare[st], 2.);
        norm += psisquare[st];
      }
      l2norm /= norm;

      double eps = std::sqrt(std::sqrt(l2norm) / double(nsweeps));

      std::discrete_distribution<int> distribution(psisquare.begin(),
                                                   psisquare.end());

      std::vector<double> countsDirect(hilb_index.NStates(), 0);
      netket::default_random_engine rgen(34251);

      for (std::size_t sw = 0; sw < nsweeps; sw++) {
        countsDirect[distribution(rgen)] += 1.;
      }

      double Z = 0;

      for (std::size_t k = 0; k < hilb_index.NStates(); k++) {
        auto X = countsDirect[k];
        auto Y = countsSampler[k];

        Z += double(std::pow(X - Y, 2.) - X - Y);
      }

      Z = std::sqrt(std::abs(Z)) / double(nsweeps);

      REQUIRE(Approx(Z).margin(5. * eps) == 0);
    }
  }
}
