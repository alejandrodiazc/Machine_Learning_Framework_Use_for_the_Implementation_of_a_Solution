# Machine_Learning_Framework_Use_for_the_Implementation_of_a_Solution


The main objective of this repository is to display the adequate implementation of a Machine Learning algorithm through a network. For this particular case, a Multilayer Perceptron (Neural Network) was implemented with the SciKit Learn library. Also, a gridsearch was programmed, in order to find the best hyperparameters that allow to optimize the metrics and predictions the model can generate.

For the whole training and testing of the Neural Network, a Palmer Archipielago Penguin database was used; this database contains information that allows to solve a classification problem, which consists in determining the species of a penguin based on certain features.

Training and testing subsets were separated, so that the generated model could be used for generating predictions and evaluated, without overfitting the data.

The MultiLayer Perceptron model implementation of the SciKit Learn allows to choose from a wide variety of hyperparameters that help refining the model; amongst them, the ones where I focused the most for the implementation are the size of the hidden layers, the activation function of the neurons, the solver, the learning rate, the mamimum number of iterations that the model can reach, and the initial learning rate. For trying out several combinations of hyperparameters, a gridsearch was performed, through the use of the SciKit Learn library. In the end, I kept the best parameters for the model, according to their accuracy score.

Once the best model was chosen, I used the testing set for generating classification predictions with the Neural Network. For testing out the performance of these predictions, the accuracy score was considered.

The score value obtained for training is 0.736, while predictions score 0.626. In the Python script, the comparison between predicted and real values can be found. 

The main Python script is called 'NeuralNetwork_Framework.py', while the database is 'penguins_size.csv'.


https://experiencia21.tec.mx/courses/315754/assignments/9858946?return_to=https%3A%2F%2Fexperiencia21.tec.mx%2Fcalendar%23view_name%3Dmonth%26view_start%3D2022-09-09
