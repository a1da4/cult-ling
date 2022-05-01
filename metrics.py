import numpy as np

def cos(vec_i_t1, vec_i_t2):
    eta = 1e-8
    inner_product = np.dot(vec_i_t1, vec_i_t2)
    cos = inner_product / (np.linalg.norm(vec_i_t1) * np.linalg.norm(vec_i_t2) + eta)
    return cos

def global_measure(vec_i_t1, vec_i_t2):
    cos_sim = cos(vec_i_t1, vec_i_t2)
    cos_dist = 1 - cos_sim
    return cos_dist

def local_measure(vecs_t1, vecs_t2, V, target_id, topn=20):
    sims_t1 = np.zeros([V])
    sims_t2 = np.zeros([V])
    for j in range(V):
        sim_t1 = cos(vecs_t1[target_id], vecs_t1[j])
        sim_t2 = cos(vecs_t2[target_id], vecs_t2[j])
        sims_t1[j] += sim_t1
        sims_t2[j] += sim_t2

    topn_t1 = np.argsort(-1 * sims_t1)[1:topn+1]
    topn_t2 = np.argsort(-1 * sims_t2)[1:topn+1]
    topn_join = list(set(topn_t1) | set(topn_t2))
    sims_topn_t1 = sims_t1[topn_join]
    sims_topn_t2 = sims_t2[topn_join]

    cos_dist = global_measure(sims_topn_t1, sims_topn_t2)
    return cos_dist
