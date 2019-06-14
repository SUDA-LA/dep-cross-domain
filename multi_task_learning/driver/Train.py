import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Parser import *
from data.Dataloader import *
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
def train(data1, data2,cw_dev_data,cdt_dev_data,cw_test_data,cdt_test_data, parser, vocab, config):
    type1, type2 = data1[0].tgt_type, data2[0].tgt_type
    optimizer = Optimizer(filter(lambda p: p.requires_grad, parser.model.parameters()), config)

    global_step = 0
    cw_best_LAS = 0
    cw_best_iter = 0
    cdt_best_LAS = 0
    cdt_best_iter = 0
    one_iter_sentence_num=config.domain_num_corpus_weighting+config.domain_num_corpus_weighting*config.cdt_domain_ration
    batch_num = int(np.ceil(one_iter_sentence_num/ float(config.train_batch_size)))
    for iter in range(config.train_iters):
        one_iter_data1, one_iter_data2 = data_corpus_weighting(data1, data2, config.domain_num_corpus_weighting,config.cdt_domain_ration)
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_arc_correct, overall_label_correct, overall_total_arcs = 0, 0, 0
        for onebatch1, onebatch2 in data_iter_from_two_data(one_iter_data1, one_iter_data2, config.train_batch_size, True):
            parser.model.train()
            loss_two = None
            if len(onebatch1) > 0:
                words1, extwords1, tags1, heads1, rels1, lengths1, masks1 = batch_data_variable(onebatch1, vocab)
                parser.forward(words1, extwords1, tags1, masks1,type1)
                loss = parser.compute_loss(heads1, rels1, lengths1)
                loss = loss / config.update_every
                loss_two = loss
            if len(onebatch2) > 0:
                words2, extwords2, tags2, heads2, rels2, lengths2, masks2 = batch_data_variable(onebatch2, vocab)
                parser.forward(words2, extwords2, tags2, masks2,type2)
                loss = parser.compute_loss(heads2, rels2, lengths2)
                loss = loss / config.update_every
                if loss_two is None:
                    loss_two = loss
                else:
                    loss_two += loss
            loss_two.backward()

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, parser.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                parser.model.zero_grad()       
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                cw_dev_arc_correct, cw_dev_rel_correct, cw_dev_arc_total, cw_dev_uas, cw_dev_las = evaluate(cw_dev_data,parser,vocab)
                cdt_dev_arc_correct, cdt_dev_rel_correct, cdt_dev_arc_total, cdt_dev_uas, cdt_dev_las = evaluate(cdt_dev_data, parser, vocab)
                print("Domain Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % (cw_dev_arc_correct, cw_dev_arc_total, cw_dev_uas, cw_dev_rel_correct, cw_dev_arc_total, cw_dev_las))
                print("CDT Dev: uas = %d/%d = %.2f, las = %d/%d =%.2f" % ( cdt_dev_arc_correct, cdt_dev_arc_total, cdt_dev_uas, cdt_dev_rel_correct, cdt_dev_arc_total,cdt_dev_las))
                if cw_dev_las > cw_best_LAS:  # 测试在test上的结果
                    cw_best_iter = iter
                    print("cw Exceed best dev las: history = %.2f, current = %.2f, [iter=%d]" % ( cw_best_LAS, cw_dev_las, cw_best_iter))
                    cw_test_arc_correct, cw_test_rel_correct, cw_test_arc_total, cw_test_uas, cw_test_las = evaluate( cw_test_data, parser, vocab)
                    cdt_test_arc_correct, cdt_test_rel_correct, cdt_test_arc_total, cdt_test_uas, cdt_test_las = evaluate(cdt_test_data, parser, vocab)
                    print("Domain Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % (cw_test_arc_correct, cw_test_arc_total, cw_test_uas, cw_test_rel_correct, cw_test_arc_total,cw_test_las))
                    print("CDT Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % (cdt_test_arc_correct, cdt_test_arc_total, cdt_test_uas, cdt_test_rel_correct,cdt_test_arc_total,cdt_test_las))
                    cw_best_LAS = cw_dev_las
                if cdt_dev_las > cdt_best_LAS:  # 测试在test上的结果
                    cdt_best_iter = iter
                    print("cdt Exceed best dev las: history = %.2f, current = %.2f, [iter=%d]" % (cdt_best_LAS, cdt_dev_las, cdt_best_iter))
                    cw_test_arc_correct, cw_test_rel_correct, cw_test_arc_total, cw_test_uas, cw_test_las = evaluate(cw_test_data, parser, vocab)
                    cdt_test_arc_correct, cdt_test_rel_correct, cdt_test_arc_total, cdt_test_uas, cdt_test_las = evaluate(cdt_test_data, parser, vocab)
                    print("Domain Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % (cw_test_arc_correct, cw_test_arc_total, cw_test_uas, cw_test_rel_correct, cw_test_arc_total,cw_test_las))
                    print("CDT Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % ( cdt_test_arc_correct, cdt_test_arc_total, cdt_test_uas, cdt_test_rel_correct,cdt_test_arc_total,cdt_test_las))
                    cdt_best_LAS = cdt_dev_las
        if iter - cw_best_iter > 100 and iter - cdt_best_iter > 100:
            print("100 no increase")
            exit(0)


def evaluate(data, parser, vocab):
    type_corpus = data[0].tgt_type
    start = time.time()
    parser.model.eval()
    #output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, tags, heads, rels, lengths, masks = \
            batch_data_variable(onebatch, vocab)
        count = 0
        arcs_batch, rels_batch = parser.parse(words, extwords, tags, lengths, masks,type_corpus)
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            #printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(predict=tree, gold=onebatch[count].sentence)
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    #output.close()

    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda' ,action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    vocab = creatVocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))



    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    cw_data = read_corpus(config.domain_file, vocab)
    cdt_data = read_corpus(config.train_file, vocab)
    cw_dev_data = read_corpus(config.cw_dev_file, vocab)
    cdt_dev_data = read_corpus(config.cdt_dev_file, vocab)
    cw_test_data = read_corpus(config.cw_test_file, vocab)
    cdt_test_data = read_corpus(config.cdt_test_file, vocab)
    tgt_type1, tgt_type2 = cw_data[0].tgt_type, cdt_data[0].tgt_type

    model = ParserModel(vocab, config, vec,tgt_type1, tgt_type2)
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    parser = BiaffineParser(model, vocab.ROOT)

    train(cw_data, cdt_data, cw_dev_data, cdt_dev_data, cw_test_data, cdt_test_data, parser, vocab, config)

