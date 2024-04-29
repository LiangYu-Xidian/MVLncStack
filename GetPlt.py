import matplotlib.pyplot as plt

class lossPlt:

    def __init__(self):
        self.plt = plt
        self.plt.figure()


    def Draw(self,list_x,list_y,filename,color,label):
        self.plt.plot(list_x, list_y, color, lw=1.5, label=label)
        self.plt.xlabel('epoch')
        self.plt.ylabel('loss')
        self.plt.title('Epoch and Loss')
        self.plt.savefig(filename)