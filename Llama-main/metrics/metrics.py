import csv
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from openpyxl import Workbook
import openpyxl
from openpyxl.utils import get_column_letter
from collections import defaultdict
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm


gt_impression=row[0]
pred_impression=row[1]
bleu_score_temp=sentence_bleu([gt_impression], pred_impression)
f1, precision, recall = calculate_lcs_f1_precision_recall_chinese(pred_impression, gt_impression)


def longest_common_subsequence_chinese(sentence1, sentence2):

    n, m = len(sentence1), len(sentence2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sentence1[i - 1] == sentence2[j - 1]: 
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[n][m]

def calculate_lcs_f1_precision_recall_chinese(predicted, reference):

    lcs_length = longest_common_subsequence_chinese(predicted, reference)
    
    precision = lcs_length / len(predicted) if predicted else 0
    recall = lcs_length / len(reference) if reference else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, precision, recall
