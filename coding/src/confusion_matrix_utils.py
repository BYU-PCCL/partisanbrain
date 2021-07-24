import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, categories):
    '''
    Plot confusion matrix.
    Arguments:
        cm (array): confusion matrix
        categories (array): list of categories
    '''
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(categories)), categories, rotation=90)
    plt.yticks(np.arange(len(categories)), categories)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def swap_indices(cm, categories, i, j):
    '''
    Swap indices of confusion matrix.
    Arguments:
        cm (array): confusion matrix
        categories (array): list of categories
        i (int): index of first category
        j (int): index of second category
    Returns:
        cm (array): confusion matrix
        categories (array): list of categories
    '''
    cm = np.array(cm)
    cm[[i, j], :] = cm[[j, i], :]
    cm[:,[i, j]] = cm[:,[j, i]]
    categories[i], categories[j] = categories[j], categories[i]
    return cm, categories

def reorder_indices(cm, categories, ordering):
    '''
    Reorder indices of confusion matrix.
    Arguments:
        cm (array): confusion matrix
        categories (array): list of categories
        ordering (array): list of indices
    Returns:
        cm (array): confusion matrix
        categories (array): list of categories
    '''
    cm = np.array(cm)
    cm = cm[ordering, :][:, ordering]
    categories = [categories[i] for i in ordering]
    return cm, categories

def sort_confusion_matrix(cm, categories):
    '''
    Sort confusion matrix according to category accuracy.
    Arguments:
        cm (array): confusion matrix
        categories (array): list of categories
    Returns:
        sorted_cm (array): sorted confusion matrix
        sorted_categories (array): list of sorted categories
    '''
    accuracies = [cm[i, i] / np.sum(cm[i, :]) for i in range(len(categories))]
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_cm, sorted_categories = reorder_indices(cm, categories, sorted_indices)
    return sorted_cm, sorted_categories


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def clustering_score(cm):
    '''
    Compute clustering score. Score is higher if more weight on off diagonals, lower otherwise.
    '''
    score = (cm[1:] * cm[:-1]).sum() + (cm[:, 1:] * cm[:, :-1]).sum()
    score += (cm[2:] * cm[:-2]).sum() + (cm[:, 2:] * cm[:, :-2]).sum()

    # METHOD 2
    n_diags = len(cm)
    diag_weight = 1 / np.arange(1, n_diags)**2

    # score = 0
    for k, weight in zip(range(n_diags), diag_weight):
        # add off-diagonal weight
        score += weight * (np.diag(cm, k) + np.diag(cm, -k)).sum()

    # TODO - try convolve?
    # kernels = [np.ones((i, i)) for i in range(2, 10)]
    # kernels = [np.ones((i, i)) for i in range(3, 8, 2)]
    # # score = 0
    # for kernel in kernels:
    #     score += np.diag(convolve2D(cm, kernel)).sum()
    
    # TODO - try k means or something?
    return score

def get_best_confusion_matrix(cm, categories):
    '''
    Get best confusion matrix according to clustering score.
    Arguments:
        cm (array): confusion matrix
        categories (array): list of categories
    Returns:
        best_cm (array): best confusion matrix
        best_categories (array): list of best categories
    '''
    # first sort by accuracy
    cm, categories = sort_confusion_matrix(cm, categories)
    score = clustering_score(cm)
    best_cm = cm
    best_categories = categories
    improved = True
    while improved:
        improved = False
        cm, categories = best_cm.copy(), best_categories.copy()
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                # swap indices
                cm, categories = swap_indices(cm, categories, i, j)
                score_ = clustering_score(cm)
                if score_ > score:
                    score = score_
                    best_cm = cm.copy()
                    best_categories = categories.copy()
                    improved = True
                # swap back
                cm, categories = swap_indices(cm, categories, i, j)
    return best_cm, best_categories

if __name__ == '__main__':
    # read in data from cm.npy
    cm = np.load('cm.npy')
    categories = ['Agriculture', 'Arts and Entertainment', 'Banking, Finance, and Domestic Commerce', 'Churches and Religion',
        'Civil Rights, Minority Issues, and Civil Liberties', 'Community Development and Housing Issues', 'Death Notices', 'Defense',
        'Education', 'Energy', 'Environment', 'Fires', 'Foreign Trade', 'Government Operations', 'Health', 'Immigration',
        'International Affairs and Foreign Aid', 'Labor', 'Law, Crime, and Family Issues', 'Macroeconomics',
        'Other, Miscellaneous, and Human Interest', 'Public Lands and Water Management', 'Social Welfare',
        'Space, Science, Technology and Communications', 'Sports and Recreation', 'State and Local Government Administration',
        'Transportation', 'Weather and Natural Disasters']

    # plot_confusion_matrix(cm, categories)

    # sort confusion matrix
    # cm, categories = sort_confusion_matrix(cm, categories)
    # plot_confusion_matrix(cm, categories)

    # # swap last and second to last
    # cm, categories = swap_indices(cm, categories, len(categories) - 1, len(categories) - 2)
    # plot_confusion_matrix(cm, categories)

    # get best
    cm, categories = get_best_confusion_matrix(cm, categories)
    plot_confusion_matrix(cm, categories)

    breakpoint()