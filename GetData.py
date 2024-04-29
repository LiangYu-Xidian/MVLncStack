import multiprocessing as mp
import sys
import Kmer, DAC, PseDNC
src_vocab = {'P': 0, 'A': 1, 'G': 2, 'C': 3, 'U': 4}



#用于存储序列和序列的标签
class SeqAndLabel():
    def __init__(self,name,seq,label):
        super(SeqAndLabel, self).__init__()
        self.name = name
        self.seq = seq
        self.label = label



#用于存储序列的机器学习特征和标签以及转换的序列
class MyDataset():
    def __init__(self, name, mlfeature, dlfeature, label):
        super(MyDataset, self).__init__()
        self.name = name
        self.mlfeature = mlfeature
        self.dlfeature = dlfeature
        self.label = label

#计算sequence-level的特征
def load_ml_feature(seq):
    feature_vector = []

    KmerDis = Kmer.make_distance_kmer(seq = seq)
    Kmer3 = Kmer.make_kmer_vector(k=3, seq=seq)
    Kmer4 = Kmer.make_kmer_vector(k=4, seq=seq)
    Kmer5 = Kmer.make_kmer_vector(k=5, seq=seq)
    feature_vector = feature_vector + KmerDis + Kmer3 + Kmer4 + Kmer5

    return feature_vector

#处理序列的信息
def Get_DL_Feature(seq,left,right):

    seq = seq.strip()
    if len(seq) >= left + right:
        seq_left = seq[: left]
        seq_right = seq[-right:]
        pos_seq = seq_left + seq_right
    else:
        pos_seq = seq.ljust(left + right, 'P')

    num_data = []
    # 用数字编码代替二级结构
    for n in pos_seq:
        num = src_vocab[n]
        num_data.append(num)
    return num_data

#将核苷酸特征和序列信息特征结合
def Get_Data_Feature(fasta_name ,seq,label,left,right,feature):

    name = fasta_name

    mlfeature = load_ml_feature(seq)
    dlfeature = Get_DL_Feature(seq=seq, left=left, right=right)

    data = MyDataset(name, mlfeature, dlfeature, label)
    return data


#多线程处理序列
class Mul_load_data():

    def __init__(self, filename, left, right, cpucore, feature):
        self.filename = filename
        self.seq_list = mp.Manager().list()
        self.data_list = mp.Manager().list()

        self.left = left
        self.right = right
        self.cpucore = cpucore

        self.feature = feature

        self.members = [i for i in range(self.cpucore)]


    def create_lists(self):
        file = open(self.filename, 'r')


        for line in file:
            if line[0] == '>':
                label = []
                seq_label = SeqAndLabel(str(line.strip().split('|')[2:]), file.readline().strip(), int(line[1]))
                self.seq_list.append(seq_label)

        return self.seq_list

    def finish_work(self,who):
        while len(self.seq_list) > 0:
            SL = self.seq_list.pop()
            data = Get_Data_Feature(SL.name, SL.seq, SL.label, self.left, self.right, self.feature)
            self.data_list.append(data)

    def start(self):
        self.create_lists()
        pool = mp.Pool(processes=self.cpucore)
        for i, member in enumerate(self.members):
            pool.apply_async(self.finish_work, (member,))

        pool.close()
        pool.join()


    #将数据返回去
    def GetData(self):

        Nu = 0
        Cyto = 0
        Ribo = 0
        Exo = 0
        for data in self.data_list:
            if data.label == 0:
                Nu = Nu + 1
            elif data.label == 1:
                Cyto = Cyto + 1
            elif data.label == 2:
                Ribo = Ribo + 1
            elif data.label == 3:
                Exo = Exo + 1
        print(self.filename + '\tNu: ' + str(Nu) + '\tCyto: ' + str(Cyto) + '\tRibo: ' + str(Ribo) + '\tExo: ' + str(Exo))
        return self.data_list
