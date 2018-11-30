/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#pragma once
#include <functional>
#include <vector>
#include <pybind11/numpy.h>
#include "featurizer.h"
#include "frame.h"
#include "fogofwar.h"

namespace py = pybind11;

// Stuff to think about featurizing...
//    Spell casters (one hot vec of dim 10)

/*
 * Input:
 *  @unit_list : vector<vector<vector<Unit>>> Batch of torchcraft Units,
 *    unit_list[batch_dim][time_step][n_units]
 *  @map_data : ??? Map data
 *
 *  @ret : pair<vector<vector<Tensor>>, vector<Tensor>>
 *    ret.first is the input, a batch of tensor features.
 *    ret.second is the target, a batch of targets.
 */
class CoarseConvFeaturizer : Featurizer {
  public:
  template<typename T> using vector = std::vector<T>;
  using Unit = torchcraft::replayer::Unit;
  using Frame = torchcraft::replayer::Frame;
  template<typename T> using Batch = vector<Frame>;
  using Feature = vector<vector<vector<double>>>;

  // Base class for reduced features
  struct Reduced {
    virtual ~Reduced() {}
  };

  size_t feature_size;
  size_t resX, resY;
  int32_t strideX, strideY;
  size_t from, until;

  CoarseConvFeaturizer(
      size_t resX = 1, size_t resY = 1,
      int32_t strideX = 0, int32_t strideY = 0,
      size_t from = 0, size_t until = 0) :
    resX(resX), resY(resY),
    strideX(strideX == 0 ? resX : strideX),
    strideY(strideY == 0 ? resY : strideY),
    from(from), until(until) { }

  virtual ~CoarseConvFeaturizer() { }

  // -1 is full perspective
  vector<Feature> featurize(
      const vector<vector<Frame*>>&,
      const vector<std::pair<size_t, size_t>>,
      int32_t perspective,
      bool full = false) const;

  virtual void featurize_unit(Feature&, Unit&, bool, bool) const = 0;

  vector<char> reduce(vector<Frame*> const& frames) const;
  vector<Feature> featurize_reduced(
      vector<char> const& data,
      size_t skip_frames,
      size_t combine_frames,
      std::pair<size_t, size_t> const& map_size,
      int32_t perspective,
      bool full = false) const;
  vector<vector<Feature>> featurize_reduced_all(
      vector<char> const& data,
      size_t skip_frames,
      size_t combine_frames,
      std::pair<size_t, size_t> const& map_size,
      int32_t perspective,
      bool visibility,
      py::array_t<uint8_t> ground_height) const;
  virtual void reduce_frame(std::ostream &out, Frame* frame) const {
    throw std::runtime_error("not implemented");
  }
  virtual Reduced* read_reduced_frame(std::istream& in) const {
    throw std::runtime_error("not implemented");
  }
  virtual Reduced* combine_reduced_frames(
      vector<CoarseConvFeaturizer::Reduced*>::const_iterator begin,
      vector<CoarseConvFeaturizer::Reduced*>::const_iterator end,
      int32_t perspective,
      bool full) const {
    throw std::runtime_error("not implemented");
  }
  virtual void featurize_reduced_frame(
      Feature& feature,
      Reduced* rframe,
      int32_t perspective,
      bool full) const {
    throw std::runtime_error("not implemented");
  }
  virtual void featurize_visibility(
      Feature& feature,
      Reduced* rframe,
      int32_t perspective) const {
    throw std::runtime_error("not implemented");
  }
  virtual void featurize_invfow(
      Feature& feature,
      Reduced* rframe,
      int32_t perspective,
      TilesInfo& tinfo,
      FogOfWar& fow) const {
    throw std::runtime_error("not implemented");
  }

  template<typename T>
  void writePOD(std::ostream& out, T const& val) const {
    out.write(reinterpret_cast<char const*>(&val), sizeof(T));
  }
  template<typename T>
  void readPOD(T& val, std::istream& in) const {
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
  }

  bool move_not_walkable = false;
  bool greedy = false;
  double div_hp = 100;
  double div_shield = 100;
  bool nonoop = true;
  double div_coord = 30;
  bool attack_out_of_range = true;
  double div_dmg = 10;
  int32_t tgt_size = 4;
  int32_t walktiles_move = 8;
  bool delta_hp = false;
  bool dense = false;
  double div_cd = 20;
  int32_t max_unit_types = 234;
  double alive_threshold = 0.5;
  bool no_moves = false;
  int32_t skip_frames = 1;
};

