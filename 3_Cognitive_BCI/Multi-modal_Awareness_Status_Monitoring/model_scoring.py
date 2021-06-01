import h5py

def normalize(matrix):
    for i in [0,1,2]:
        matrix[i]=matrix[i]/sum(matrix[i])
    return matrix


class CVSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, s, i, batch_size, data, data_source):
        self.x = x
        self.y = y
        self.s = s
        self.i = i
        self.data = data
        self.data_source = data_source
        self.batch_size = batch_size

        if not self.data:   # self.data could be 0(False) or not 0(True). so this means if self.data=False
            self.input_shape = self.data_source['data']['00']['ppg'][self.x[0][0]:self.x[0][1]].shape
        else:
            self.input_shape = self.data[self.s[0]][self.x[0][0]:self.x[0][1]].shape
        self.output_shape = (self.y.shape[1],)
        np.random.shuffle(self.i)
    def __getitem__(self, index=None):
        if index is None:
            index = self.i
        else:
            index = self.i[index * self.batch_size:(index + 1) * self.batch_size]
        if not self.data:
            x = np.stack(list(map(lambda i: self.data_source['data'][f'{self.s[i]:02}']['ppg'][self.x[i][0]:self.x[i][1],0], index)))
        else:
            x = np.stack(list(map(lambda i: self.data[self.s[i]][self.x[i][0]:self.x[i][1],0], index)))
        y = np.stack(list(map(lambda i: self.y[i], index)))
        return x, y
    def __len__(self):
        return int(np.floor(len(self.i) / self.batch_size))
    def on_epoch_end(self):
        np.random.shuffle(self.i)
