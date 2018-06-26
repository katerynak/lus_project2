from subprocess import call, check_output
import pandas as pd
from data_elaboration import load
from embeddings import vocabulary, create_w2id


def write_pred_result(x, y_true, y_pred, predFileName):
    """
    function writes to file sentences, predicted results and true labels
    :param x: sentences
    :param y_pred: predicted tags
    :param y_true: correct tags
    :param predFileName: out file name
    :return:
    """

    pred_data = pd.DataFrame([x, y_true, y_pred])
    pred_data = pred_data.transpose()
    pred_data.to_csv(predFileName, index=None, header=None, sep=' ', mode='w')


def evaluate(predFileName, evalFileName):
    """
    given predFileName where prediction results are stored, function calls evaluation script and redirects
    output to evalFileName
    :param predFileName:
    :param evalFileName:
    :return: f1 score
    """

    call('./conlleval.pl < {0} > {1}'.format(predFileName, evalFileName), shell=True)
    f1_score = check_output("awk '{print $8}' " + "{0} |sed '2q;d'".format(evalFileName), shell=True).decode("utf-8")[:-2]
    accuracy = check_output("awk '{print $2}' " + "{0} |sed '2q;d'".format(evalFileName), shell=True).decode("utf-8")[:-3]
    precision = check_output("awk '{print $4}' " + "{0} |sed '2q;d'".format(evalFileName), shell=True).decode("utf-8")[:-3]
    recall = check_output("awk '{print $6}' " + "{0} |sed '2q;d'".format(evalFileName), shell=True).decode("utf-8")[:-3]
    return accuracy, precision, recall, f1_score
