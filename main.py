from __future__ import division
from __future__ import print_function
from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from pyod.models.cblof import CBLOF
from pyod.utils.data import generate_data
from pyod.utils.data import get_color_codes
from pyod.utils.data import evaluate_print


def visualize(clf_name, X_train, y_train, X_test,y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=False):  # pragma: no cover
    """
    Utility function for visualizing the results in examples
    Internal use only

    :param clf_name: The name of the detector
    :type clf_name: str

    :param X_train: The training samples
    :param X_train: numpy array of shape (n_samples, n_features)

    :param y_train: The ground truth of training samples
    :type y_train: list or array of shape (n_samples,)

    :param X_test: The test samples
    :type X_test: numpy array of shape (n_samples, n_features)

    :param y_test: The ground truth of test samples
    :type y_test: list or array of shape (n_samples,)

    :param y_train_pred: The predicted outlier scores on the training samples
    :type y_train_pred: numpy array of shape (n_samples, n_features)

    :param y_test_pred: The predicted outlier scores on the test samples
    :type y_test_pred: numpy array of shape (n_samples, n_features)

    :param show_figure: If set to True, show the figure
    :type show_figure: bool, optional (default=True)

    :param save_figure: If set to True, save the figure to the local
    :type save_figure: bool, optional (default=False)
    """

    if X_train.shape[1] != 3 or X_test.shape[1] != 3:
        raise ValueError("Input data has to be 2-d for visualization. The "
                         "input data has {shape}.".format(shape=X_train.shape))

    X_train, y_train = check_X_y(X_train, y_train)
    X_test, y_test = check_X_y(X_test, y_test)
    c_train = get_color_codes(y_train)
    c_test = get_color_codes(y_test)

    fig = plt.figure()
    plt.suptitle("Demo of {clf_name}".format(clf_name=clf_name))

    ax1 = fig.add_subplot(221, projection='3d')

    ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=c_train)  #生成散点
    plt.title('Train ground truth')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='normal',
                              markerfacecolor='b', markersize=6),
                       Line2D([0], [0], marker='o', color='w', label='outlier',
                              markerfacecolor='r', markersize=6)]
    plt.legend(handles=legend_elements, loc=4)#图例

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=c_test)
    plt.title('Test ground truth')
    plt.legend(handles=legend_elements, loc=4)

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train_pred)
    plt.title('Train prediction by {clf_name}'.format(clf_name=clf_name))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='normal',
                              markerfacecolor='0', markersize=6),
                       Line2D([0], [0], marker='o', color='w', label='outlier',
                              markerfacecolor='yellow', markersize=6)]
    plt.legend(handles=legend_elements, loc=4)

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test_pred)
    plt.title('Test prediction by {clf_name}'.format(clf_name=clf_name))
    plt.legend(handles=legend_elements, loc=4)

    if save_figure:
        plt.savefig('{clf_name}.png'.format(clf_name=clf_name), dpi=2400)
    if show_figure:
        plt.show()
    return


def ROC(X_test, y_test, y_test_scores, n, con):
    X_test, y_test = check_X_y(X_test, y_test)
    y_scores = y_test_scores
    scores_max = max(y_test_scores)
    # maxp = math.sqrt(scores_max)
    k = len(y_scores) - 1
    print(k)
    c = 50
    threshold = [0]*c
    scores = [0]*(k+1)
    TPR = [0]*c
    FPR = [0]*c
    NUMALL = [0]*c
    for num in range(0, c):
        threshold[num] = scores_max*0.02*(num)
        TP = 1
        FP = 0
        NUM = 1
        for i in range(0, k):
            if y_scores[i] >= threshold[num]:
                scores[i] = 1
                NUM = NUM+1
            else:
                scores[i] = 0

            if y_test[i] == 1 and scores[i] == 1:
                TP = TP+1
            if y_test[i] == 0 and scores[i] == 1:
                FP = FP+1
        NUMALL[num] = NUM
        print(NUMALL)
        TPR[num] = TP/(n*con)
        print(TPR)
        FPR[num] = FP/(n*(1-con))
        print(FPR)
    f = FPR
    t = TPR
    plt.plot(f, t)
    plt.scatter(f, t, c='r', marker='o')
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    plt.plot(NUMALL, TPR)
    plt.scatter(NUMALL, TPR, c='r', marker='o')
    plt.title('detectable rate')
    plt.xlabel('NUM')
    plt.ylabel('TPR')
    plt.show()
    return


if __name__ == "__main__":
    contamination = 0.1  # percentage of outliers
    n_train = 600  # number of training points
    n_test = 200  # number of testing points
    # Generate sample data
    X_train, y_train, X_test, y_test = \
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=3,
                      contamination=contamination,
                      random_state=42)
    print(X_train)
    # train CBLOF detector
    clf_name = 'CBLOF'
    clf = CBLOF()
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores
    print(y_train_pred)
    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores
    print(max(y_test_scores))
    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)

    ROC(X_test, y_test, y_test_scores, n_test, contamination)
    # visualize the results
    visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)



