from data_elaboration import load, transform, Wrapped_dataset, load_words
from embeddings import vocabulary, create_w2id, doc2ids, ids2doc
from LSTMTagger import LSTMTagger
from GRU import GRU
from RNN import RNN
import torch
from pred_evaluation import evaluate, write_pred_result
from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from filter_w2v import get_w2v, get_w2v_w2id
import argparse
import numpy as np


def data_prep(filename):
    print(filename)
    data = load(filename)

    # we first create word to idx then we create tag to idx model
    word_vocab = vocabulary(data.tokens)
    word2id = create_w2id(word_vocab)
    tag_vocab = vocabulary(data.tags)
    tag2id = create_w2id(tag_vocab)

    return data, word2id, tag2id


def train(model, epochs, dataloader, loss_function, optimizer, trainfile, devfile, testfile, word2id, tag2id, pred_file="", eval_file=""):
    max_f1 = 0
    count_dec = 0
    epoch_num = 0
    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        epoch_num = epoch
        for batch in dataloader:
            model.zero_grad()
            tag_scores, target_scores = model(batch)
            loss = loss_function(tag_scores, target_scores)
            loss.backward()
            optimizer.step()
        accuracy, precision, recall, f1score = test(model, word2id, tag2id, trainfile, pred_file, eval_file)
        print("train accuracy: {}, precision: {}, recall: {}, f1score: {}".format(accuracy, precision, recall, f1score))
        model.eval()
        accuracy, precision, recall, f1score = test(model, word2id, tag2id, devfile, pred_file, eval_file)
        print("dev accuracy: {}, precision: {}, recall: {}, f1score: {}".format(accuracy, precision, recall, f1score))
        if float(f1score) > max_f1:
            max_f1 = float(f1score)
            count_dec = 0
        else:
            count_dec += 1
        model.train()
        if count_dec > 5:
            break
    model.eval()
    pred_file += "__actual_epochs_{}".format(epoch_num)
    eval_file += "__actual_epochs_{}".format(epoch_num)
    accuracy, precision, recall, f1score = test(model, word2id, tag2id, testfile, pred_file, eval_file)
    print("test accuracy: {}, precision: {}, recall: {}, f1score: {}".format(accuracy, precision, recall, f1score))

    return model


def test(model, word2id, tag2id, filename="../data/NLSPARQL.test.data", pred_file=None, eval_file=None):
    """
    :param model: trained model
    :param filename: test filename
    :return:
    """

    data, _, _ = data_prep(filename)
    data_transform = partial(transform, sentence_transform=partial(doc2ids, word2id=word2id),
                             tags_transform=partial(doc2ids, word2id=tag2id))

    data_wrapped = Wrapped_dataset(data, data_transform)
    padded_size = len(data_wrapped[0]['tags'])

    dataloader = DataLoader(data_wrapped, batch_size=40,
                            shuffle=False, num_workers=4, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x)

    predicted_out = []

    with torch.no_grad():
        for batch in dataloader:
            tag_scores, _ = model(batch)
            predicted_out.append(tag_scores.argmax(dim=1))

    predicted_out = torch.cat(predicted_out).view(-1, padded_size)
    predicted_out_tags = []
    for predicted_tags, input_sentence in zip(predicted_out, data.tokens):
        predicted_out_tags.append(ids2doc(predicted_tags[:len(input_sentence.split())], tag2id))
        predicted_out_tags.append([float('nan')])

    predicted_out_tags = [item for sublist in predicted_out_tags for item in sublist]

    #evaluation results directory for evaluation script results
    # eval_dir = "../eval_out/rnn/"
    # if not os.path.exists(eval_dir):
    #     os.makedirs(eval_dir)

    data_words = load_words(filename)
    write_pred_result(data_words.x.tolist(), data_words.y_true.tolist(), predicted_out_tags, pred_file)
    accuracy, precision, recall, f1score = evaluate(pred_file, eval_file)
    return accuracy, precision, recall, f1score


def train_evaluate(modelarc, trainfile, devfile, testfile, batch_size, lr, decay, emb, embedding_size, freeze, drop,
                   epochs, hidden_size, word_unk_prob=0, tag_unk_prob=0):
    torch.manual_seed(1)

    outfiledir = "../models_out/{}/".format(modelarc)
    if not os.path.exists(outfiledir):
        os.makedirs(outfiledir)

    outfile = "batch_size_{}__lr_{}__decay_{}__emb_{}__embedding_size_{}__freeze_{}__" \
              "drop_{}__epochs_{}__hidden_size_{}__word_unk_prob_{}__tag_unk_prob_{}".format(batch_size, lr, decay, emb, embedding_size, freeze, drop, epochs,
                                                       hidden_size, word_unk_prob, tag_unk_prob)
    if os.path.isfile(outfiledir + outfile):
        return

    print(modelarc)
    print(outfile)

    pred_dir = "../pred_data/{}/".format(modelarc)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    eval_dir = "../eval_out/{}/".format(modelarc)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("loading file {}".format(trainfile))

    # data preparation
    data, word2id, tag2id = data_prep(trainfile)
    w2v_values = None
    pretrained = False
    if emb == 'pretrained':
        print("pretrained")
        pretrained = True
        w2vfile = "../embeddings/w2v_trimmed.pickle"
        w2v = get_w2v(w2vfile, trainfile)
        word2id = get_w2v_w2id(w2v)
        w2v_values = w2v.values

    data_transform = partial(transform, sentence_transform=partial(doc2ids, word2id=word2id, unk_probability=word_unk_prob),
                             tags_transform=partial(doc2ids, word2id=tag2id, unk_probability=tag_unk_prob))

    data_wrapped = Wrapped_dataset(data, data_transform)
    dataloader = DataLoader(data_wrapped, batch_size=batch_size,
                            shuffle=False, num_workers=4, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x)

    model = None

    if modelarc == 'lstm':
        model = LSTMTagger(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                           vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1,
                           drop_rate=drop, freeze=freeze)

    if modelarc == 'bilstm':
        model = LSTMTagger(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                           vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1,
                           drop_rate=drop, freeze=freeze)

    if modelarc == 'gru':
        model = GRU(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1,
                    drop_rate=drop, freeze=freeze)

    if modelarc == 'bigru':
        model = GRU(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1,
                    drop_rate=drop, freeze=freeze)

    if modelarc == 'rnn':
        model = RNN(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1,
                    drop_rate=drop, freeze=freeze, jordan=False)

    if modelarc == 'birnn':
        model = RNN(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1,
                    drop_rate=drop, freeze=freeze, jordan=False)

    if modelarc == 'jordan':
        model = RNN(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1,
                    drop_rate=drop, freeze=freeze, jordan=True)

    if modelarc == 'bijordan':
        model = RNN(embedding_size, hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1,
                    drop_rate=drop, freeze=freeze, jordan=True)

    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print("total trainable parameters %i" % total_parameters)

    loss_function = partial(torch.nn.functional.nll_loss, ignore_index=-1)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, amsgrad=True,
                                 weight_decay=decay)

    model = train(model, epochs, dataloader, loss_function, optimizer, trainfile=trainfile,
                  devfile=devfile, testfile=testfile, word2id=word2id, tag2id=tag2id, pred_file=pred_dir + outfile, eval_file=eval_dir + outfile)

    torch.save(model.state_dict(), outfiledir + outfile)


if __name__== "__main__":
    models = ["lstm", "bilstm", "gru", "bigru", "rnn", "birnn", "jordan", "bijordan"]
    embeddings = ["pretrained", "train"]

    parser = argparse.ArgumentParser(description='script creates and trains tagging model passed as argument.')

    parser.add_argument('--model', dest='model', metavar='model_name', type=str, required=True,
                        choices=models,
                   help='model to train, possible models are :{}'.format(models))

    parser.add_argument('--trainfile', dest='trainfile', metavar='train_file_name', type=str, required=True,
                    help='filename of train data')

    parser.add_argument('--devfile', dest='devfile', metavar='dev_file_name', type=str, default=None,
                        help='filename of dev data')

    parser.add_argument('--testfile', dest='testfile', metavar='test_file_name', type=str, default=None,
                        help='filename of test data')

    parser.add_argument('--batch', dest='batch_size', metavar='batch_size', type=int, default=20,
                        help='batch size dimension')

    parser.add_argument('--lr', dest='lr', metavar='learning_rate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--emb', dest='embeddings', metavar='word_embeddigns', type=str, default='pretrained',
                        choices=embeddings,
                        help='embeddings to use: {}, default: pretrained'.format(embeddings))

    parser.add_argument('--drop', dest='dr', metavar='drop_rate', type=float, default=0.5,
                        help='drop rate for drop out layers')

    parser.add_argument('--epochs', dest='epochs', metavar='epochs_number', type=int, default=20,
                        help='number of epochs')

    parser.add_argument('--hidden', dest='hidden_size', metavar='hidden_size', type=int, default=200,
                        help='hidden size for lstm models')

    parser.add_argument('--emb_size', dest='embedding_size', metavar='embedding_size', type=int, default=300,
                        help='size for embeddings')

    parser.add_argument('--freeze', dest='freeze', metavar='freeze_bool', type=bool, default=False,
                        help='set to True if freezing of embeddings is preferred')

    parser.add_argument('--out', dest='modelfile', metavar='model_filename', type=str, default='../models_out/model',
                        help='filename to save model')

    args = parser.parse_args("--model lstm --trainfile ../data/NLSPARQL.train --devfile "
                             "../data/NLSPARQL.dev --batch 20 --lr 0.001 --emb pretrained".split())

    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("loading file {}".format(args.trainfile))

    # data preparation
    data, word2id, tag2id = data_prep(args.trainfile)
    w2v_values = None
    pretrained=False
    if args.embeddings=='pretrained':
        pretrained=True
        w2vfile = "../embeddings/w2v_trimmed.pickle"
        w2v = get_w2v(w2vfile, args.trainfile)
        word2id = get_w2v_w2id(w2v)
        w2v_values = w2v.values

    data_transform = partial(transform, sentence_transform=partial(doc2ids, word2id=word2id),
                             tags_transform=partial(doc2ids, word2id=tag2id))

    data_wrapped = Wrapped_dataset(data, data_transform)
    dataloader = DataLoader(data_wrapped, batch_size=20,
                            shuffle=False, num_workers=4, drop_last=False, pin_memory=True,
                            collate_fn=lambda x: x)

    model = None

    if args.model=='lstm':
        model = LSTMTagger(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                 vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1, drop_rate=args.dr, freeze=args.freeze)

    if args.model=='bilstm':
        model = LSTMTagger(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                 vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1, drop_rate=args.dr, freeze=args.freeze)

    if args.model=='gru':
        model = GRU(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                 vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1, drop_rate=args.dr, freeze=args.freeze)

    if args.model=='bigru':
        model = GRU(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                 vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1, drop_rate=args.dr, freeze=args.freeze)

    if args.model == 'rnn':
        model = RNN(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1,
                    drop_rate=args.dr, freeze=args.freeze, jordan=False)

    if args.model == 'birnn':
        model = RNN(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1,
                    drop_rate=args.dr, freeze=args.freeze, jordan=False)

    if args.model == 'jordan':
        model = RNN(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=False, num_layers=1,
                    drop_rate=args.dr, freeze=args.freeze, jordan=True)

    if args.model == 'bijordan':
        model = RNN(args.embedding_size, args.hidden_size, len(tag2id), device, pretrained_embeddings=pretrained,
                    vocab_size=len(word2id), w2v_weights=w2v_values, bidirectional=True, num_layers=1,
                    drop_rate=args.dr, freeze=args.freeze, jordan=True)

    model.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print("total trainable parameters %i" % total_parameters)

    loss_function = partial(torch.nn.functional.nll_loss, ignore_index=-1)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True,
                                 weight_decay=0)

    model = train(model, args.epochs, dataloader, loss_function, optimizer, trainfile=args.trainfile,
                  devfile=args.devfile, testfile="../data/NLSPARQL.test.data",
                  word2id=word2id, tag2id=tag2id, pred_file="", eval_file="")

    torch.save(model.state_dict(), args.modelfile)


