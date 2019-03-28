from fastai.text import *

import fire


def load_model(itos_filename, classifier_filename, num_classes):
    """Load the classifier and int to string mapping
    Args:
        itos_filename (str): The filename of the int to string mapping file (usually called itos.pkl)
        classifier_filename (str): The filename of the trained classifier
    Returns:
        string to int mapping, trained classifer model
    """

    # load the int to string mapping file
    itos = pickle.load(Path(itos_filename).open('rb'))
    # turn it into a string to int mapping (which is what we need)
    stoi = collections.defaultdict(lambda:0, {str(v):int(k) for k,v in enumerate(itos)})

    # these parameters aren't used, but this is the easiest way to get a model
    bptt,em_sz,nh,nl = 70,400,1150,3
    dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5
    vs = len(itos)

    model = get_rnn_classifier(bptt, 20*70, num_classes, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
            layers=[em_sz*3, 50, num_classes], drops=[dps[4], 0.1],
            dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    # load the trained classifier
    model.load_state_dict(torch.load(classifier_filename, map_location=lambda storage, loc: storage))

    # put the classifier into evaluation mode
    model.reset()
    model.eval()

    return stoi, model


def softmax(x):
    '''
    Numpy Softmax, via comments on https://gist.github.com/stober/1946926
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    '''
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def predict_text(stoi, model, text):
    """Do the actual prediction on the text using the
        model and mapping files passed
    """

    # prefix text with tokens:
    #   xbos: beginning of sentence
    #   xfld 1: we are using a single field here
    input_str = 'xbos xfld 1 ' + text

    # predictions are done on arrays of input.
    # We only have a single input, so turn it into a 1x1 array
    texts = [text]
    print(texts)
    # tokenize using the fastai wrapper around spacy
    # tok = Tokenizer().proc_all_mp(partition_by_cores(texts))

    # turn into integers for each word
    encoded = [stoi[p] for p in texts]

    # we want a [x,1] array where x is the number
    #  of words inputted (including the prefix tokens)
    ary = np.reshape(np.array(encoded),(-1,1))

    # turn this array into a tensor
    tensor = torch.from_numpy(ary)

    # wrap in a torch Variable
    variable = Variable(tensor)

    # do the predictions
    predictions = model(variable)

    # convert back to numpy
    numpy_preds = predictions[0].data.numpy()

    return softmax(numpy_preds[0])[0]


def predict_input(itos_filename, trained_classifier_filename, num_classes=10):
    """
    Loads a model and produces predictions on arbitrary input.
    :param itos_filename: the path to the id-to-string mapping file
    :param trained_classifier_filename: the filename of the trained classifier;
                                        typically ends with "clas_1.h5"
    :param num_classes: the number of classes that the model predicts
    """
    # Check the itos file exists
    if not os.path.exists(itos_filename):
        print("Could not find " + itos_filename)
        exit(-1)

    # Check the classifier file exists
    if not os.path.exists(trained_classifier_filename):
        print("Could not find " + trained_classifier_filename)
        exit(-1)

    stoi, model = load_model(itos_filename, trained_classifier_filename, num_classes)


    text = '7436 10734 13163 10244 13646 3701 1037 8435 13160 1918 1906 10291 1243 10647 11707 12031 1172 9627 1189 811 ' \
           '11243 12113 6131 11247 6970 12836 5924 6269 10653 11368 8465 1144 3224 12050 13422 14272 3122 4036 770 ' \
           '10968 10435 2498 11702 10234 773 866 875 2231 14997 13997 767 10360 11233 1148 14365 2728 12190 7002 4112 ' \
           '12083 9572 9833 7023 14324 769 13978 8574 12179 6948 12100 10355 12856 10507 10454 9613 5401 8051 3214 14948' \
           ' 8408 11702 3433 11084 8112 14394 11140 2489 3295 4092 11396 3677 12785 1149 1851 8170 4147 10517 1145'
    scores = predict_text(stoi, model, text)
    print("Result id {0}, Scores: {1}".format(np.argmax(scores), scores))


if __name__ == '__main__':
    fire.Fire(predict_input)