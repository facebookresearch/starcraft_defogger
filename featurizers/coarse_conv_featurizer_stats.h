/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#pragma once
#include "coarse_conv_featurizer.h"

// This macro corresponds to the size of each feature
// Can only be used within this class due to skip frames
#define CoarseConvFeaturizerStats_DO_FEATURES(F) \
F(N_UNITS,     1)  /* How many units are on this tile, mostly for flyers */ \
F(VELOCITY_X,  1)  /* float, velocity on x direction */ \
F(VELOCITY_Y,  1)  /* float, velocity on y direction */ \
F(HEALTH,      1)  /* float, health of the unit */ \
F(SHIELD,      1)  /* float, health of the unit */ \
F(ARMOR,       1)  /* float, armor of the unit */ \
F(TARGET_X,    1)  /* float, x position of the target of the order */ \
F(TARGET_Y,    1)  /* float, y position of the target of the order */ \
F(TARGET_DIST, 1)  /* float, distance to target (divided by div_coord) */ \
F(GWATTACK,    1)  /* float, ground weapon attack damage (divided by div_dmg) */ \
F(GWDMGTYPE,   6)  /* one hot, ground weapon damage type. can be Independent, Explosive, */ \
                    /* Concussive, Normal, Ignore_armor, or None */ \
F(GWRANGE,     1)  /* float, range of the ground attack (divided by div_coord) */ \
F(AWATTACK,    1)  /* float, air weapon attack damage (divided by div_dmg) */ \
F(AWDMGTYPE,   6)  /* one hot, air weapon damage type (same as ground) */ \
F(AWRANGE,     1)  /* float, range of the air attack (divided by div_coord) */ \
F(N_FLYING,    1)   /* number of flyers */ \
F(N_BUILDING,  1)   /* number of flyers */ \



class CoarseConvFeaturizerStats : public CoarseConvFeaturizer {
  public:
  CoarseConvFeaturizerStats(
      size_t resX = 1, size_t resY = 1,
      int32_t strideX = 0, int32_t strideY = 0,
      size_t from = 0, size_t until = 0);
  virtual ~CoarseConvFeaturizerStats() { }

#define DECLARE_FEATURES(NAME, SIZE) uint32_t NAME; uint8_t NAME##_size;
  CoarseConvFeaturizerStats_DO_FEATURES(DECLARE_FEATURES);
#undef DECLARE_FEATURES

  virtual void featurize_unit(Feature&, Unit&, bool, bool) const override;
};
