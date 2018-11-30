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
#include "featurizer.h"
#include "frame.h"

// This macro corresponds to the size of each feature
// Can only be used within this class due to skip frames
#define ForwardConv2DStateActionFeaturizer_DO_FEATURES(F) \
F(IS_ENEMY,    1)  /* {-1,1) whether the unit is enemy */ \
F(CD,          1)  /* float, unit cool down (divided by div_cd) */ \
F(VELOCITY_X,  1)  /* float, velocity on x direction */ \
F(VELOCITY_Y,  1)  /* float, velocity on y direction */ \
F(IS_EVALUATING,1) /* {-1,1) whether we are deciding action for this unit/order */ \
F(IS_UNIT,     1)  /* boolean, 1 if this is a unit (0 if order) */ \
F(MAXCD,       1)  /* float, max cool down of the unit */ \
F(ALLY_HEALTH, 1)  /* float, health of the unit if it is an ally */ \
F(ENEMY_HEALTH,1)  /* float, health of the unit if it is an enemy */ \
F(ARMOR,       1)  /* float, armor of the unit */ \
F(IS_ORDER,    1)  /* boolean, whether this is an order */ \
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
F(ORDER_FRAMES, skip_frames + 1) /* one hot, frames during which order was active. */ \
F(IS_EVALUATING_ORD,  1) /* boolean, whether order is being evaluated */ \
F(IS_EVALUATED_ORD,   1) /* boolean, whether order has been evaluated */ \
F(IS_ORDER_TARGET,    1) /* boolean, whether the feature correspond to the target of an order */ \
/* All the features below that point correspond to reflection */ \
/*of the features of the unit giving the order (source).      */ \
F(IS_SOURCE_ENEMY,    1) \
F(SOURCE_X,           1) \
F(SOURCE_Y,           1) \
F(SOURCE_DIST,        1) \
F(SOURCE_CD,          1) \
F(SOURCE_GWATTACK,    1) \
F(SOURCE_GWDMGTYPE,   6) \
F(SOURCE_GWRANGE,     1) \
F(SOURCE_AWATTACK,    1) \
F(SOURCE_AWDMGTYPE,   6) \
F(SOURCE_AWRANGE,     1) \
F(SOURCE_VELOCITY_X,  1) \
F(SOURCE_VELOCITY_Y,  1) \
F(ORDER_FRAMES_TARGET, skip_frames + 1) \
F(SOURCE_IS_EVALUATING,        1) \
F(SOURCE_IS_EVALUATING_ORD,    1) \
F(SOURCE_IS_EVALUATED_ORD,     1)

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
class ForwardConv2DStateActionFeaturizer : Featurizer {
  public:
  template<typename T> using Vector = std::vector<T>;
  using Unit = torchcraft::replayer::Unit;
  using UnitList = Vector<Unit>;
  using Sample = std::pair<UnitList, UnitList>;
  template<typename T> using Batch = Vector<Vector<T>>;
#define DECLARE_FEATURES(NAME, SIZE) uint32_t NAME; uint8_t NAME##_size;
  ForwardConv2DStateActionFeaturizer_DO_FEATURES(DECLARE_FEATURES);
#undef DECLARE_FEATURES

  uint32_t feature_size;

  struct Feature {
    int32_t x, y, our_type, our_order, their_type, their_order;
    Vector<double> feats;
  };

  ForwardConv2DStateActionFeaturizer();

  Batch<Feature> featurize(Batch<Sample> &batch);

  void featurize_unit(Vector<Feature>&, Unit&, bool, std::function<Unit*(int32_t)>);

  Vector<Feature> featurize_single(Sample &state);

  Batch<Feature> featurize_batch_input(Vector<Sample> &batch);

  bool move_not_walkable = false;
  bool greedy = false;
  double div_hp = 100;
  bool nonoop = true;
  double div_coord = 30;
  bool attack_out_of_range = true;
  double div_dmg = 10;
  int32_t tgt_size = 4;
  int32_t walktiles_move = 8;
  bool delta_hp = false;
  bool dense = false;
  double div_cd = 20;
  // double tgt_ind = table;
  int32_t max_unit_types = 234;
  double alive_threshold = 0.5;
  bool no_moves = false;
  // ind = table;
  int32_t skip_frames = 1;
};
