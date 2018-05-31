from subprocess import call, check_output
import pandas as pd


def write_pred_result(x, y_pred, y_true, predFileName):

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
    f1_score = check_output("awk '{print $8}' " + "{0} |sed '2q;d'".format(evalFileName), shell=True).decode("utf-8")
    return f1_score


#TODO: ADD CUSTOM PARAMETER SEARCH FROM EVALUATION FILE


def test():
    return


if __name__ == "__main__":
    """
    arguments : test file, architecture, 
    """