import Generate
import numpy
import tensorflow
import time
import copy

# ===============================
# ====== [Hyperparameters] ======
# ===============================

num_feature = 20
num_label = 2
knowledgeability = 1
iteration = 100
hypo_matrix = Generate.Boundary_Hypo_Table(num_feature, True)
num_hypo = len(hypo_matrix)
ptxy = 1 / num_feature / num_label

# ===============================
# ====== [Numpy Matrix] =========
# ====== [Memory Usage Note] ====
# ====== [1000 Features] ========
# ====== [About 300 MB RAM] =====
# ===============================

# PYXH Matrix
P_y_xh = numpy.empty((num_label, num_feature, num_hypo), dtype="float32")

# Knowledgeability Matrix
Delta_g_h = numpy.zeros((num_hypo, num_hypo), dtype="float32")

# PLHXY Matrix
P_learner_h_xy = numpy.empty(
    (num_label, num_feature, num_hypo), dtype="float32")

# PTXYH Matrix
p_teacher_xy_h = numpy.empty(
    (num_label, num_feature, num_hypo), dtype="float32")

# PTXH Matrix
p_teacher_xh = numpy.empty(
    (num_label, num_feature, num_hypo), dtype="float32")

# PHX Matrix
P_hx = numpy.empty((num_label, num_feature, num_hypo), dtype="float32")

# ====================================
# ====== [Initialize Variables] ======
# ====================================

for h in range(num_hypo):
    for x in range(num_feature):
        for y in range(num_label):
            P_y_xh[y, x, h] = 1.0 if hypo_matrix[h, x] == y else 0.0

Delta_g_h.fill((1 - knowledgeability) / (num_hypo - 1) if num_hypo > 1 else 1)
numpy.fill_diagonal(Delta_g_h, knowledgeability)

P_hx.fill(1 / num_hypo)

# ==================================
# ====== [Tensorflow Session] ======
# ==================================

session = tensorflow.Session()

# The Knowledgeability Matrix
K = tensorflow.placeholder(tensorflow.float32, shape=(num_hypo, num_hypo))

# The P_Y_XH Matrix
PYXH = tensorflow.placeholder(
    tensorflow.float32, shape=(num_label, num_feature, num_hypo))
PHX = tensorflow.placeholder(
    tensorflow.float32, shape=(num_label, num_feature, num_hypo))
PTXH = tensorflow.placeholder(
    tensorflow.float32, shape=(num_label, num_feature, num_hypo))

# The Initial Step
INIT_PYXH_TIMES_PHX = tensorflow.multiply(
    PYXH, tensorflow.divide(PHX, num_hypo))

# The Teaching Step (First part)
TEACHING_PLHXY_TIMES_PTXY = tensorflow.multiply(PYXH, ptxy)

# The Teaching Normalization Step
# Add some tiny number in case of 0 normalization
TEACHING_NORMALIZATION_OVER_X_Y = tensorflow.divide(
    PYXH, tensorflow.add(tensorflow.reduce_sum(PYXH, axis=(0, 1), keep_dims=True), 1e-20))

SCALE_ALL_ELEMENTS = tensorflow.multiply(
    PYXH, tensorflow.constant(num_label, dtype=tensorflow.float32))

# The Learning Step: p_yxh[y, x, h] * phx[h] * p_teacher_xh[x, h]
LEARNING_PYXH_PHX_PTXH = tensorflow.multiply(
    tensorflow.multiply(PYXH, PHX), PTXH)

# The Learning Normalization Step
LEARNING_NORMALIZATION_OVER_H = tensorflow.divide(
    PYXH, tensorflow.add(tensorflow.reduce_sum(PYXH, axis=2, keep_dims=True), 1e-20))

SUM_HYPO = tensorflow.reduce_sum(PYXH, axis=2)
SUM_LABEL = tensorflow.reduce_sum(PYXH, axis=0)
SUM_FEATURE_AND_LABEL = tensorflow.reduce_sum(PYXH, axis=(0, 1))

session.run(tensorflow.global_variables_initializer())


# ===============================
# ====== [Running Session] ======
# ====== [The Iteration] ========
# ===============================

def Learning_Steps():
    global num_feature, num_label, knowledgeability, iteration, hypo_matrix, num_hypo, ptxy
    global P_y_xh, Delta_g_h, P_learner_h_xy, p_teacher_xy_h, p_teacher_xh, P_hx

    begin = time.process_time()

    P_learner_h_xy = session.run(INIT_PYXH_TIMES_PHX, feed_dict={
        PYXH: P_y_xh, PHX: P_hx})

    P_learner_h_xy = session.run(
        LEARNING_NORMALIZATION_OVER_H, feed_dict={PYXH: P_learner_h_xy})

    iter = 0
    while iter < iteration:
        p_teacher_xy_h = session.run(
            TEACHING_PLHXY_TIMES_PTXY, feed_dict={PYXH: P_learner_h_xy})

        p_teacher_xy_h = session.run(
            TEACHING_NORMALIZATION_OVER_X_Y, feed_dict={PYXH: p_teacher_xy_h})

        Sum_over_y = session.run(SUM_LABEL, feed_dict={
            PYXH: p_teacher_xy_h})

        for y in range(num_label):
            for h in range(num_hypo):
                for x in range(num_feature):
                    sum = 0
                    for g in range(num_hypo):
                        sum += Sum_over_y[x, g] * Delta_g_h[g, h]
                    p_teacher_xh[y, x, h] = sum

        p_teacher_xh = session.run(TEACHING_NORMALIZATION_OVER_X_Y, feed_dict={
            PYXH: p_teacher_xh})
        p_teacher_xh = session.run(SCALE_ALL_ELEMENTS, feed_dict={
            PYXH: p_teacher_xh})

        P_learner_h_xy = session.run(LEARNING_PYXH_PHX_PTXH, feed_dict={
            PYXH: P_y_xh, PHX: P_hx, PTXH: p_teacher_xh})

        P_learner_h_xy = session.run(LEARNING_NORMALIZATION_OVER_H,
                                     feed_dict={PYXH: P_learner_h_xy})

        iter += 1

#    print("\n\nP_learner_h_xy", P_learner_h_xy, "p_teacher_xy_h", p_teacher_xy_h,
#          "p_teacher_xh", p_teacher_xh, sep="\n\n")

    print("Iteration Finished, Processing Time: ", time.process_time() - begin)

# Learning_Steps()

# ===============================
# ====== [Running Session] ======
# ====== [The Task] =============
# ===============================


# Get the feature with the highest Pt
# observable_feature_set: the set that contains all select-able features
# hypo: current true hypo
def Get_Feature(observable_feature_set, hypo_idx):
    global p_teacher_xh

    max_idx = observable_feature_set[0]
    max_value = p_teacher_xh[0, max_idx, hypo_idx]
    for feature in observable_feature_set:
        if p_teacher_xh[0, feature, hypo_idx] > max_value:
            max_value = p_teacher_xh[0, feature, hypo_idx]
            max_idx = feature
    return max_idx

# Observe the target feature when there is a true hypothesis
# hypo_map: the set of all hypothesis
# true_hypo: the true hypo
# target_feature: the target features we want to observe
# return:
#      the probability of finding the true hypothesis
#      the label of the feature


def Observe(true_hypo_idx, target_feature_idx):
    global hypo_matrix, P_learner_h_xy

    list = []
    # Get the true label
    true_hypo = hypo_matrix[true_hypo_idx]
    true_label = true_hypo[target_feature_idx]

    for hypo in hypo_matrix:
        if hypo[target_feature_idx] != true_label:
            check = False
        else:
            list.append(hypo)
    return P_learner_h_xy[true_label, target_feature_idx, true_hypo_idx], true_label

# hypo_map: The map of the hypothesis
# return: a map from hypothesis to observation * probability


def Probability_Task():
    global num_feature, num_label, knowledgeability, iteration, hypo_matrix, num_hypo, ptxy
    global P_y_xh, Delta_g_h, P_learner_h_xy, p_teacher_xy_h, p_teacher_xh, P_hx

    prob_map = {}
    select_map = {}
    const_feature_set = []

    # Append all observable features to the feature set
    for f in range(num_feature):
        const_feature_set.append(f)

    # Assume there is a true hypo = hypo
    # Get all posible hypothesis in the hypo map
    for hypo_idx in range(num_hypo):

        # Make a copy to save time
        feature_set = copy.deepcopy(const_feature_set)

        prob = []
        select = []

        while True:
            Learning_Steps()

            # Pick the feature with the highest PT
            feature = Get_Feature(feature_set, hypo_idx)

            prob_find, true_label = Observe(hypo_idx, feature)
            prob.append(prob_find)
            select.append(feature)

            # Assign the p_learner_h_xy to phx
            for x in range(num_feature):
                for y in range(num_label):
                    P_hx[y, x, :] = P_learner_h_xy[true_label, feature, :]

            # remove the feature in the feature set,
            # make the same feature only be observed once
            feature_set.remove(feature)

            if (len(feature_set) == 0) or (prob_find >= 1.0):
                prob_map[hypo_idx] = prob
                select_map[hypo_idx] = select
                P_hx.fill(1 / num_hypo)
                break
    return prob_map, select_map

p_map, s_map = Probability_Task()
print(p_map,s_map,sep="\n")

session.close()
