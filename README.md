# PSL-DL
Method Development for Predicting Protein Subcellular Localization Based on Deep Learning


# PSL_WCLA
In view of the highly nonlinear, unequal length of protein sequences, a novel method for protein subcellular localization prediction based on distributed coding and convolutional cycle self-attention mechanism is proposed. Firstly, the word vectors were trained by unsupervised learning with sequence data from protein database, and then the nonlinear features were extracted by convolutional neural network and long short-term memory network (LSTM).In order to comprehensively consider the long sequence feature information, the self-attention mechanism is added to learn the global feature. In addition, considering the problems of resources and training efficiency, the time convolutional neural network is designed to replace LSTM, which solves the problem of insufficient utilization of resources. 

# PSL_DA
Aiming at the problem of large amount of unlabeled data in current protein database, a prediction method of protein subcellular localization based on conditional antagonistic network domain adaptation was studied. Firstly, the source domain data and the target domain data are defined, and then the common feature representation between the data in different domains is learned according to the network model.

# PSL_ulmfit
In order to avoid learning from scratch for sequences with different data distributions, a prediction method of protein subcellular localization based on language model was proposed. Finally, the prediction results on the SWISS-PROT data show the effectiveness and advancement of the two strategies.

# PSL_GAN
An algorithm of protein sequence generation based on feedback generation antagonistic network model is proposed to solve the problem that protein sequence data cannot be generated effectively at specific subcellular localization sites. First, the idea of reinforcement learning (Policy Gradient) was used to solve the problem of difficult reverse propagation in protein sequence generation. Then, in order to ensure the high quality of generated sequences, a method of real-time feedback generation was proposed.
