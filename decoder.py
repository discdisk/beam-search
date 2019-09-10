import numpy as np
import chainer.functions as F




class Beam_search(object):
    """docstring for Beam_search"""
    def __init__(self, n_vocab : int, vocab : dir, beamWidth : int):
        super(Beam_search, self).__init__()
        self.n_vocab = n_vocab
        self.vocab = vocab
        self.beamWidth = beamWidth

        self.beam = {}

        self.texts = ['']
        self.probs = [1]
        self.last_words = ['_blank']

    def sort(self, mat:np.array, beamWidth:int) -> list:
        # sort to two part by kth largest element (smaller->kth->larger)
        # larger part and smaller part don't have order
        sorted_index = np.argpartition(mat, len(mat)-beamWidth)
        return sorted_index [-beamWidth:]

    def merge(self, texts, probs, last_words) -> None:
        # merge those paths which have save transcript and previous word
        new_texts = []
        new_probs = []
        new_last_words = []
        search_dic = {}
        count=0


        for text,prob,last_word in zip(texts, probs, last_words):
            if (text,last_word) not in search_dic:
                search_dic[(text,last_word)] = count

                new_texts.append(text)
                new_probs.append(0)
                new_last_words.append(last_word)
                count+=1

            index = search_dic[(text,last_word)]
            new_probs[index]+=prob

        return new_texts, new_probs, new_last_words

    def add(self, frame:np.array) -> None:
        #add new step of data into the beams

        # use chainer's softmax function
        # need to replace with numpy implementation
        frame=F.softmax(frame,0).data 

        new_texts = []
        new_probs = []
        new_last_words = []


        sorted_frame_indexes = self.sort(frame, self.beamWidth)
        # print(sorted_frame_indexes)

        for text,prob,last_word in zip(self.texts, self.probs, self.last_words):
            for i in sorted_frame_indexes:
                new_text=text

                curr_word = self.vocab[i]
                curr_prob = frame[i]

                if curr_word != last_word and curr_word != '_blank':
                    new_text+=' '
                    new_text+=curr_word
                new_texts.append(new_text)
                new_probs.append(prob*curr_prob)
                new_last_words.append(curr_word)

        # print(sorted_frame_indexes)
        new_texts, new_probs, new_last_words = self.merge(new_texts, new_probs, new_last_words)

        if len(new_probs)>self.beamWidth:
            sorted_path_indexes = self.sort(new_probs, self.beamWidth)
        else:
            sorted_path_indexes = [i for i in range(len(new_probs))]

        self.texts = np.array(new_texts)[sorted_path_indexes]
        self.probs = np.array(new_probs)[sorted_path_indexes]
        self.last_words = np.array(new_last_words)[sorted_path_indexes]



    def getBest(self) -> str:
        # print(self.probs)
        i = np.argmax(self.probs)
        print(self.texts[i][1:])
        # print(self.probs[i])

        return self.texts[i][1:]

        

if __name__ == '__main__':
    import editdistance
    import pickle
    vocab = pickle.load(open('word_dic.pkl','rb'))
    vocab = {y:x for x,y in vocab.items()}
    # print(vocab)
    # prob_matrixs = pickle.load(open('prob_matrix.pkl','rb'))
    # pickle.dump(prob_matrix,open('test_data.pkl','wb'))
    # exit()
    prob_matrixs = pickle.load(open('test_data.pkl','rb'))
    all_wer=0

    for prob_matrix in prob_matrixs:
        searcher = Beam_search(len(vocab), vocab, beamWidth=100)
        mat=prob_matrix[0]
        print(mat.shape)
        
        for i,frame in enumerate(mat):
            searcher.add(frame)

        ground_truth = ' '.join([vocab[x] for x in prob_matrix[1]])
        predict = searcher.getBest()

        wer=editdistance.eval(ground_truth.split(' '), predict.split(' '))/len(ground_truth.split(' '))

        all_wer+=wer

        print('WER is :',wer)

        print(ground_truth)

    print(all_wer/len(prob_matrixs))
