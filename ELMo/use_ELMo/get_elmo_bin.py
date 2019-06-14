
import struct
from allennlp.modules.elmo import Elmo,batch_to_ids

class get_elmo:
    def __init__(self,options_file,weight_file,test_file,outfile):
        self.options_file=options_file
        self.weight_file = weight_file
        self.test_file = test_file
        self.outfile = outfile
    def get_elmo(self):
        test_file = open(self.test_file, "r")
        outfile = open(self.outfile, "w")
        elmo = Elmo(self.options_file, self.weight_file, 2, dropout=0)
        sentence = []
        sen=[]
        sentencenum = 0
        for line in test_file:
            if line == "\n":
                sentencenum += 1
                print(sentencenum,flush=True)
                sen.append(sentence)
                character_ids = batch_to_ids(sen)
                print(character_ids)
                embeddings = elmo(character_ids)
                embs=embeddings['elmo_representations'][0]
                for char in range(embs.shape[1]):
                    for layer in range(embs.shape[0]):
                        for dim in range(embs.shape[2]):
                            outfile.write(str(float(embs[layer][char][dim]))+' ')
                        outfile.write('#')
                    outfile.write('\n')
                outfile.write("\n")
                outfile.flush()

                sentence = []
                sen=[]
            else:
                sentence.append(line.strip().split('\t')[1])
        test_file.close()
        outfile.close()
class write_bin:
    def __init__(self,inputfile,outputfile):
        self.inputfile = inputfile
        self.outputfile = outputfile
    def get_lines(self):
        f1 = open(self.inputfile, "r")
        word = 0
        sens = 0
        count = 0
        for line in f1:
            count += 1
            if line != '\n' and line != '\r\n':
                word += 1
            else:
                sens += 1
        print("count", count)
        print("word", word)
        print("sens", sens)

    def get_first_layer(self):
        infile = open(self.inputfile, "r")
        outfile = open(self.outputfile, "wb")

        file_embs = []
        sentence_emb_first = []
        sentence_emb_second = []
        sentence_emb_third = []
        sentencenum = 0
        for line in infile:
            if line != '\n':            
                parts = line.strip().split('#')
                firstlayer = parts[0].split()
                if len(firstlayer) != 1024:
                    print("first layer", len(firstlayer))

                sentence_emb_first.append(firstlayer)
            else:
                sentencenum += 1
                for first in sentence_emb_first:
                    for emb1 in first:
                        emb = float(emb1)
                        outfile.write(struct.pack('f', float(emb)))
                if sentencenum % 1000 == 0:
                    print(sentencenum, flush=True)
                sentence_emb_first = []
                sentence_emb_second = []
                sentence_emb_third = []
        print('sentencenum', sentencenum)
if __name__ == "__main__":
    train_file = "./train.conll"
    train_outfile = "./train/train"
 
    A = get_elmo("./options.json","./weights.hdf5",train_file,train_outfile+".elmo")
    A.get_elmo()
    B = write_bin(train_outfile+".elmo",train_outfile+".1024.bin")
    B.get_lines()
    B.get_first_layer()
    

