# Machine_Learning_Framework_Use_for_the_Implementation_of_a_Solution
https://experiencia21.tec.mx/courses/315754/assignments/9858946?return_to=https%3A%2F%2Fexperiencia21.tec.mx%2Fcalendar%23view_name%3Dmonth%26view_start%3D2022-09-09


The main objective of this repository is to display the adequate implementation of a Machine Learning algorithm through a network. For this particular case, a Multilayer Perceptron (Neural Network) was implemented with the SciKit Learn library. Also, a gridsearch was programmed, in order to find the best hyperparameters that allow to optimize the metrics and predictions the model can generate.

For the whole training and testing of the Neural Network, a Palmer Archipielago Penguin database was used; this database contains information that allows to solve a classification problem, which consists in determining the species of a penguin based on certain features.

Training and testing subsets were separated, so that the generated model could be used for generating predictions and evaluated.

The metric that was chosen for evaluation of the models is the accuracy score, which can be easily obtained with a function in SciKit Learn library.

The score value obtained for training is 0.736, while predictions score 0.626

The main Python script is called 'NeuralNetwork_Framework.py', while the database is 'penguins_size.csv'.
