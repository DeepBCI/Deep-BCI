import h5py

def normalize(matrix):
    for i in [0,1,2]:
        matrix[i]=matrix[i]/sum(matrix[i])
    return matrix


def get_score(conf_mat, method='accuracy'):
    a_real = conf_mat[0]
    s_real = conf_mat[1]
    d_real = conf_mat[2]
    a_pred = conf_mat.T[0]
    s_pred = conf_mat.T[1]
    d_pred = conf_mat.T[2]

    if method == 'recall':
        rec_a = a_real[0] / sum(a_real)
        rec_s = s_real[1] / sum(s_real)
        rec_d = d_real[2] / sum(d_real)
        return (rec_a+rec_s+rec_d)/3
    if method == 'precision':
        prc_a = a_pred[0] / sum(a_pred)
        prc_s = s_pred[1] / sum(s_pred)
        prc_d = d_pred[2] / sum(d_pred)
        return (prc_a+prc_s+prc_d)/3
    else:
        corr = a_real[0] + s_real[1] + d_real[2]
        return corr / sum(sum(conf_mat))
