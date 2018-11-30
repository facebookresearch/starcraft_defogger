/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#include "featurizers/forward_conv2d_state_action_featurizer.h"
#include "featurizers/coarse_conv_featurizer_stats.h"
#include "featurizers/coarse_conv_featurizer_units.h"
#include "frame.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

template<typename T> using vector = std::vector<T>;
using Unit = torchcraft::replayer::Unit;
using Frame = torchcraft::replayer::Frame;
namespace py = pybind11;

template <typename T>
py::array_t<double> featurize(
    const T *self,
    const vector<vector<Frame*>> &batch,
    const std::pair<size_t, size_t> map_sizes,
    int32_t perspective,
    bool full) {
  vector<CoarseConvFeaturizer::Feature> batch_features;
  {
    py::gil_scoped_release release;
    vector<std::pair<size_t, size_t>> sizes(batch.size(), map_sizes);
    batch_features = self->featurize(batch, sizes, perspective, full);
  }
  py::array_t<double> result({
      (size_t)(batch_features.size()),
      (size_t)(batch_features[0].size()),
      (size_t)(batch_features[0][0].size()),
      (size_t)(2 * self->feature_size)
      });
  auto data = result.mutable_unchecked<4>();
  // This _should_ be safe but it might not be...
  py::gil_scoped_release release;
  for (size_t bid = 0; bid < batch_features.size(); bid++) {
    auto &carr = batch_features[bid];
    for (size_t i = 0; i < carr.size(); i++)
      for (size_t j = 0; j < carr[0].size(); j++)
        for (size_t k = 0; k < (size_t)(2 * self->feature_size); k++)
          data(bid, i, j, k) = carr[i][j][k];
  }
  return result;
}

template <typename T>
py::array_t<double> featurize_reduced(
    T const* self,
    py::bytes const& rdata,
    size_t skip_frames,
    size_t combine_frames,
    std::pair<size_t, size_t> const map_size,
    int32_t perspective,
    bool full) {
  vector<CoarseConvFeaturizer::Feature> batch_features;
  {
    py::gil_scoped_release release;
    // XXX these copies are driving me nuts
    std::string str = rdata;
    std::vector<char> vdata(str.begin(), str.end());
    batch_features = self->featurize_reduced(
        vdata, skip_frames, combine_frames, map_size, perspective, full);
  }
  py::array_t<double> result({
      (size_t)(batch_features.size()),
      (size_t)(batch_features[0].size()),
      (size_t)(batch_features[0][0].size()),
      (size_t)(2 * self->feature_size)
      });
  auto data = result.mutable_unchecked<4>();
  // This _should_ be safe but it might not be...
  py::gil_scoped_release release;
  for (size_t bid = 0; bid < batch_features.size(); bid++) {
    auto &carr = batch_features[bid];
    for (size_t i = 0; i < carr.size(); i++)
      for (size_t j = 0; j < carr[0].size(); j++)
        for (size_t k = 0; k < (size_t)(2 * self->feature_size); k++)
          data(bid, i, j, k) = carr[i][j][k];
  }
  return result;
}

template <typename T>
std::vector<py::array_t<double>> featurize_reduced_all(
    T const* self,
    py::bytes const& rdata,
    size_t skip_frames,
    size_t combine_frames,
    std::pair<size_t, size_t> const map_size,
    int32_t perspective,
    bool visibility,
    py::array_t<uint8_t> ground_height) {
  vector<vector<CoarseConvFeaturizer::Feature>> batch_features;
  {
    py::gil_scoped_release release;
    // XXX these copies are driving me nuts
    std::string str = rdata;
    std::vector<char> vdata(str.begin(), str.end());
    batch_features = self->featurize_reduced_all(
        vdata, skip_frames, combine_frames, map_size, perspective, visibility,
        ground_height);
  }
  vector<py::array_t<double>> result;
  for (auto& batch : batch_features) {
    if (batch.size() == 0) continue;
    auto fsize = batch[0][0][0].size();
    result.push_back(py::array_t<double>({batch.size(),
      batch[0].size(),
      batch[0][0].size(),
      fsize,
    }));
    // TODO: assert that features have the same size
    auto data = result.back().mutable_unchecked<4>();
    // This _should_ be safe but it might not be...
    py::gil_scoped_release release;
    for (size_t bid = 0; bid < batch.size(); bid++) {
      auto &carr = batch[bid];
      for (size_t i = 0; i < carr.size(); i++)
        for (size_t j = 0; j < carr[0].size(); j++)
          for (size_t k = 0; k < fsize; k++) {
            data(bid, i, j, k) = carr[i][j][k];
          }
    }
  }
  return result;
}

template <typename T>
py::bytes reduce(
    T const* self,
    vector<Frame*> const& batch) {
  vector<char> data;
  {
    py::gil_scoped_release release;
    data = self->reduce(batch);
  }
  return py::bytes(data.data(), data.size());
}

PYBIND11_MODULE(_ext, m) {
  py::class_<CoarseConvFeaturizer> ccfeat(m, "CoarseConvFeaturizer");
  ccfeat
    .def_readonly("resX", &CoarseConvFeaturizer::resX)
    .def_readonly("resY", &CoarseConvFeaturizer::resY)
    .def_readonly("strideX", &CoarseConvFeaturizer::strideX)
    .def_readonly("strideY", &CoarseConvFeaturizer::strideY)
    .def_readonly("feature_size", &CoarseConvFeaturizer::feature_size)
    .def("featurize", &featurize<CoarseConvFeaturizer>,
      py::arg("batch"),
      py::arg("map_sizes"),
      py::arg("perspective") = 0,
      py::arg("full") = false
    )
    .def("featurize_reduced", &featurize_reduced<CoarseConvFeaturizer>,
      py::arg("rdata"),
      py::arg("skip_frames"),
      py::arg("combine_frames"),
      py::arg("map_size"),
      py::arg("perspective") = 0,
      py::arg("full") = false
    )
    .def("featurize_reduced_all", &featurize_reduced_all<CoarseConvFeaturizer>,
      py::arg("rdata"),
      py::arg("skip_frames"),
      py::arg("combine_frames"),
      py::arg("map_size"),
      py::arg("perspective") = 0,
      py::arg("visibility") = false,
      py::arg("ground_height") = py::array_t<uint8_t>()
    )
    .def("reduce", &reduce<CoarseConvFeaturizer>,
      py::arg("batch")
    );

  py::class_<CoarseConvFeaturizerStats>(m, "CoarseConvFeaturizerStats", ccfeat)
    .def(py::init<>())
    .def(py::init<size_t, size_t>())
    .def(py::init<size_t, size_t, int32_t, int32_t>())
    .def(py::init<size_t, size_t, int32_t, int32_t, int32_t, int32_t>())
#define FEATURE_INDEX(NAME, SIZE) .def_readonly(#NAME, &CoarseConvFeaturizerStats::NAME)
  CoarseConvFeaturizerStats_DO_FEATURES(FEATURE_INDEX)
#undef FEATURE_INDEX
  ;

  py::class_<CoarseConvFeaturizerUnitTypes>(m, "CoarseConvFeaturizerUnitTypes", ccfeat)
    .def(py::init<>())
    .def(py::init<size_t, size_t>())
    .def(py::init<size_t, size_t, int32_t, int32_t>())
    .def(py::init<size_t, size_t, int32_t, int32_t, int32_t, int32_t>())
    .def_readonly("typemapper", &CoarseConvFeaturizerUnitTypes::typemapper);

  py::class_<ForwardConv2DStateActionFeaturizer>(m, "ForwardConv2DStateActionFeaturizer")
    .def(py::init<>())
#define FEATURE_INDEX(NAME, SIZE) .def_readonly(#NAME, &ForwardConv2DStateActionFeaturizer::NAME)
  ForwardConv2DStateActionFeaturizer_DO_FEATURES(FEATURE_INDEX)
#undef FEATURE_INDEX
    .def("featurize",
        [](ForwardConv2DStateActionFeaturizer *self, vector<vector<std::pair<vector<Unit>, vector<Unit>>>> &batch) {
          auto batch_features = self->featurize(batch);
          vector<std::pair<py::array_t<int>, py::array_t<double>>> result;
          for (auto features : batch_features) {
            // x, y, our_type, our_order, their_type, their_order
            auto ids = py::array_t<int>({features.size(), 6LU});
            auto feats = py::array_t<double>({features.size(), (long unsigned) self->feature_size});
            auto ids_data = ids.mutable_unchecked<2>();
            auto feats_data = feats.mutable_unchecked<2>();
            for (size_t i = 0; i < features.size(); i++) {
              ids_data(i, 0) = features[i].x;
              ids_data(i, 1) = features[i].y;
              ids_data(i, 2) = features[i].our_type;
              ids_data(i, 3) = features[i].our_order;
              ids_data(i, 4) = features[i].their_type;
              ids_data(i, 5) = features[i].their_order;
              for (size_t j=0; j<self->feature_size; j++)
                feats_data(i, j) = features[i].feats[j];
            }
            result.push_back(std::make_pair(
                  std::move(ids),
                  std::move(feats)
                  ));
          }
          return result;
        });
}
