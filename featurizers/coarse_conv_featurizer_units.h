/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#pragma once
#include <array>
#include "coarse_conv_featurizer.h"
#include "coarse_conv_featurizer_units_generated.h"

class CoarseConvFeaturizerUnitTypes : public CoarseConvFeaturizer {
  public:
    struct ReducedUnitTypes : public Reduced {
      // TODO: The ffbs data below is used for loading frames from buffers, and
      // the map of vectors is used when combining frames. Use ffbs everywhere.
      struct Unit {
        int32_t id, x, y, type, visible;
      };
      // Indexed by player
      std::unordered_map<int32_t, vector<Unit>> units;

      ffbs::ReducedUnitTypes const* d = nullptr;
      std::vector<char> data;
    };

    std::array<int,234> typemapper;
    CoarseConvFeaturizerUnitTypes(
        size_t resX = 1, size_t resY = 1,
        int32_t strideX = 0, int32_t strideY = 0,
        size_t from = 0, size_t until = 0);
    virtual ~CoarseConvFeaturizerUnitTypes() { }

    virtual void featurize_unit(Feature&, Unit&, bool, bool) const override;
    virtual void reduce_frame(std::ostream& out, Frame* frame) const override;
    virtual Reduced* read_reduced_frame(std::istream& in) const override;
    virtual Reduced* combine_reduced_frames(
        vector<CoarseConvFeaturizer::Reduced*>::const_iterator begin,
        vector<CoarseConvFeaturizer::Reduced*>::const_iterator end,
        int32_t perspective,
        bool full) const override;
    virtual void featurize_reduced_frame(
        Feature& feature,
        Reduced* rframe,
        int32_t perspective,
        bool full) const override;
    virtual void featurize_visibility(
        Feature& feature,
        Reduced* rframe,
        int32_t perspective) const override;
    virtual void featurize_invfow(
        Feature& feature,
        Reduced* rframe,
        int32_t perspective,
        TilesInfo& tinfo,
        FogOfWar& fow) const override;
    void inc_feature(Feature& feature, int32_t c, int32_t x, int32_t y) const;
    void mark_feature_vline(Feature& feature, int32_t c, int32_t x, int32_t y1,
        int32_t y2) const;

    uint8_t const rfVersion = 1;
};

