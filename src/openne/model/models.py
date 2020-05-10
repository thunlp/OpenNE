import torch
import os
from time import time

class BaseModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

        for i in kwargs:
            self.__setattr__(i, kwargs[i])

    def forward(self, dataset):
        raise NotImplementedError


class ModelWithEmbeddings(BaseModel):
    def __init__(self, *, output=None, save=True, **kwargs):
        if save:
            if output == None:
                outputpath = os.path.join('result', type(self).__name__)
                outputmodelfile = type(self).__name__ + '_model.txt'
                outputembeddingfile = type(self).__name__ + '_embeddings.txt'
            else:
                outputpath = os.path.dirname(output)
                if outputpath == '':
                    outputpath = '.'
                outputmodelfile = type(self).__name__ + 'model.txt'
                outputembeddingfile = os.path.basename(output)
            if not os.path.isdir(self.outputpath):
                os.mkdir(self.outputpath)
            self.embeddings = None
            try:
                with open(os.path.join(self.outputpath, self.outputembeddingfile), 'a') as fout:
                    pass
            except Exception as e:
                raise FileNotFoundError('Failed to open target embedding file "{}": {}. '.format(
                    os.path.join(self.outputpath, self.outputembeddingfile), str(e)))
            try:
                with open(os.path.join(self.outputpath, self.outputembeddingfile), 'a') as fout:
                    pass
            except Exception as e:
                raise FileNotFoundError('Failed to open target model file "{}": {}. '.format(
                    os.path.join(self.outputpath, self.outputembeddingfile), str(e)))

            super(ModelWithEmbeddings, self).__init__(outputpath=outputpath,
                                                      outputmodelfile=outputmodelfile,
                                                      outputembeddingfile=outputembeddingfile,
                                                      save=save,
                                                      **kwargs)
        else:
            super(ModelWithEmbeddings, self).__init__(save=save, **kwargs)


    def get_train(self, graph, **kwargs):
        raise NotImplementedError

    def save_model(self):
        filename = os.path.join(self.outputpath, self.outputmodelfile)
        torch.save(self.state_dict(), filename)

    def save_embeddings(self):
        filename = os.path.join(self.outputpath, self.outputembeddingfile)
        with open(filename, 'w') as fout:
            node_num = len(self.vectors)
            fout.write("{} {}\n".format(node_num, self.rep_size))
            for node, vec in self.vectors.items():
                fout.write("{} {}\n".format(node, ' '.join([str(float(x)) for x in vec])))

    def load(self, path=None):
        if not os.path.isfile(path):
            raise FileNotFoundError("Model file not found.")
        self.load_state_dict(torch.load(path))

    def forward(self, graph, **kwargs):
        t1 = time()
        print("Start training...")
        self.embeddings = self.get_train(graph, **kwargs)
        if self.save:
            print("Saving embeddings...")
            self.save_embeddings()
            print("Saving model...")
            self.save_model()
        t2 = time()
        print("Finished training. Time used = {}.".format(t2 - t1))
        return self.embeddings
