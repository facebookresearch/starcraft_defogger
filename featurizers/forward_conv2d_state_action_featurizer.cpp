/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 **/

#include "forward_conv2d_state_action_featurizer.h"

template<typename T> using Vector = std::vector<T>;
using Unit = torchcraft::replayer::Unit;
using Order = torchcraft::replayer::Order;
using UnitList = Vector<Unit>;
using Sample = std::pair<UnitList, UnitList>;
template<typename T> using Batch = Vector<Vector<T>>;
using Feature = ForwardConv2DStateActionFeaturizer::Feature;

/*
// Auxillary info per unit
struct UnitInfo {
    int32_t uid;
    const Unit* u;
    bool is_enemy;
    Vector<const Order*> orders;
};

UnitInfo getUnitInfo(const Unit &u, bool is_enemy, UnitList allies, UnitList enemies) {
    UnitInfo newunit;
    newunit.uid = u.id;
    newunit.u = &u;
    newunit.is_enemy = is_enemy;

    for (auto order : u.orders) {
        if (newunit.orders.size() != 0
                or !(order == u.orders[u.orders.size()-1]))
            newunit.orders.push_back(&order);
    }
}
*/

ForwardConv2DStateActionFeaturizer::ForwardConv2DStateActionFeaturizer() {
    uint32_t size = 0;
#define SET_INDS(NAME, SIZE) NAME = size; NAME##_size = SIZE; size += SIZE;
    ForwardConv2DStateActionFeaturizer_DO_FEATURES(SET_INDS);
#undef SET_INDS
    feature_size = size;
}

Vector<Feature> ForwardConv2DStateActionFeaturizer::featurize_single(Sample &state) {
    auto myself = state.first;
    auto enemy = state.second;

    std::unordered_map<int32_t, Unit*> unitmap;
    for (auto u : myself) unitmap[u.id] = &u;
    auto get_unit = [&unitmap] (int32_t uid) { return unitmap[uid]; };
    Vector<Feature> feats;
    for (auto unit : myself) featurize_unit(feats, unit, false, get_unit);
    for (auto unit : enemy) featurize_unit(feats, unit, true, get_unit);

    return feats;
}

Batch<Feature> ForwardConv2DStateActionFeaturizer::featurize_batch_input(Vector<Sample> &batch) {
    Batch<Feature> feat_batch;
    for (auto sample : batch) feat_batch.push_back(featurize_single(sample));
    return feat_batch;
}

/*
 * Input:
 *  @unit_list : vector<vector<pair<vector<Unit>, vector<Unit>>>> Batch of torchcraft Units,
 *    unit_list[batch_dim][time_step][n_units]
 *  @map_data : ??? Map data
 *
 *  @ret : pair<vector<vector<Tensor>>, vector<Tensor>>
 *    ret.first is the input, a batch of tensor features.
 *    ret.second is the target, a batch of targets.
 */
Batch<Feature> ForwardConv2DStateActionFeaturizer::featurize(Batch<Sample> &batch) {
    // Featurize first time step of each batch
    Vector<Sample> curState;
    for (auto t : batch) curState.push_back(t[0]);
    auto input_features = featurize_batch_input(curState);
    //TODO do targets too
    return input_features;
}

void ForwardConv2DStateActionFeaturizer::featurize_unit(
        Vector<Feature> &feats,
        Unit &u,
        bool is_enemy,
        std::function<Unit*(int32_t)> get_unit_fn) {
    feats.emplace_back();
    auto feat = &feats.back();
    feat->feats.resize(feature_size);

    // Add unit features on tile of unit
    feat->x = u.x; feat->y = u.y;
    feat->our_type = u.type;
    feat->feats[IS_UNIT] = 1;
    feat->feats[IS_ENEMY] = is_enemy ? 1 : -1;
    feat->feats[CD] = u.groundCD / div_cd;
    feat->feats[MAXCD] = u.maxCD / div_cd;
    feat->feats[ARMOR] = u.armor;
    feat->feats[VELOCITY_X] = u.velocityX;
    feat->feats[VELOCITY_Y] = u.velocityY;
    if (is_enemy) feat->feats[ENEMY_HEALTH] = (u.health + u.shield) / div_hp;
    else feat->feats[ALLY_HEALTH] = (u.health + u.shield) / div_hp;
    feat->feats[IS_EVALUATING] = -1; // Never true for forward model

    auto first_frame = u.orders[0].first_frame;
    auto last_frame = first_frame + skip_frames - 1;
    if (!is_enemy) last_frame++;

    for (size_t i = 0; i < u.orders.size(); i++) {
        auto order = u.orders[i];
        // Skip duplicate orders
        if (i != 0 and order == u.orders[u.orders.size()-1]) continue;

        auto dX = order.targetX - u.x;
        auto dY = order.targetY - u.y;
        auto dist = sqrt(dX*dX + dY*dY);
        auto o_last_frame = (i == u.orders.size() - 1)
            ? last_frame
            : u.orders[i+1].first_frame - 1;
        auto typ = u.type + 1;
        auto otyp = order.type + 1;

        feats.emplace_back(); feat = &feats.back();
        feat->feats.resize(feature_size);
        feat->x = u.x; feat->y = u.y;
        feat->our_type = typ;
        feat->our_order = otyp;

        feat->feats[IS_ORDER]    = 1;
        feat->feats[IS_ENEMY]    = is_enemy ? 1 : -1;
        feat->feats[CD]          = u.groundCD / div_cd;
        feat->feats[VELOCITY_X]  = u.velocityX;
        feat->feats[VELOCITY_Y]  = u.velocityY;
        feat->feats[TARGET_X]    = (order.targetId != -1) ? dX / div_coord : 0;
        feat->feats[TARGET_Y]    = (order.targetId != -1) ? dY / div_coord : 0;
        feat->feats[GWATTACK]    = u.groundATK / div_dmg;
        feat->feats[GWDMGTYPE + u.groundDmgType] = 1;
        feat->feats[GWRANGE]     = u.groundRange / div_coord;
        feat->feats[AWATTACK]    = u.airATK / div_dmg;
        feat->feats[AWDMGTYPE + u.airDmgType] = 1;
        feat->feats[AWRANGE]     = u.airRange / div_coord;
        feat->feats[TARGET_DIST] = (order.targetId != -1) ? dist : 0;
        for (int k = order.first_frame - first_frame;
                k < o_last_frame - first_frame + 1;
                k++) feat->feats[ORDER_FRAMES + k] = 1;
        // Never true for forward models TODO Change for policy models
        feat->feats[IS_EVALUATING]     = -1;
        feat->feats[IS_EVALUATING_ORD] = -1;
        feat->feats[IS_EVALUATED_ORD]  = -1;

        if (order.targetId == -1) continue;
        // Only if we have a target

        feats.emplace_back(); feat = &feats.back();
        feat->feats.resize(feature_size);
        feat->x                        = order.targetX;
        feat->y                        = order.targetY;
        feat->their_type               = typ;
        feat->their_order              = otyp;
        feat->feats[IS_ORDER_TARGET]   = 1;
        feat->feats[IS_SOURCE_ENEMY]   = is_enemy ? 1 : -1;
        feat->feats[SOURCE_X]          = dX / div_coord;
        feat->feats[SOURCE_Y]          = dY / div_coord;
        feat->feats[SOURCE_DIST]       = dist;
        feat->feats[SOURCE_CD]         = u.groundCD / div_cd;
        feat->feats[SOURCE_GWATTACK]   = u.groundATK / div_dmg;
        feat->feats[SOURCE_GWDMGTYPE + u.groundDmgType] = 1;
        feat->feats[SOURCE_GWRANGE]    = u.groundRange / div_coord;
        feat->feats[SOURCE_AWATTACK]   = u.airATK / div_dmg;
        feat->feats[SOURCE_AWDMGTYPE + u.airDmgType] = 1;
        feat->feats[SOURCE_AWRANGE]    = u.airRange / div_coord;
        feat->feats[SOURCE_VELOCITY_X] = u.velocityX;
        feat->feats[SOURCE_VELOCITY_Y] = u.velocityY;
        for (int k = order.first_frame - first_frame;
                k < o_last_frame - first_frame + 1;
                k++) feat->feats[ORDER_FRAMES + k] = 1;
        // Never true for forward models TODO Change for policy models
        feat->feats[SOURCE_IS_EVALUATING]      = -1;
        feat->feats[SOURCE_IS_EVALUATING_ORD]  = -1;
        feat->feats[SOURCE_IS_EVALUATED_ORD]   = -1;
    }
}
