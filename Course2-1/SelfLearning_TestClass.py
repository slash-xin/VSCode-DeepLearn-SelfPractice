import numpy as np
import matplotlib.pyplot as plt
from reg_utils import load_2D_dataset, plot_decision_boundary, predict_dec
from SelfLearning_DNNClassWithRegularization import DnnWithRegularizationBinaryClassifierClass

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()
plt.show()


model = DnnWithRegularizationBinaryClassifierClass()
model.InitializeTrainData(train_X, train_Y)

# Non-regularization Model
model.InitializeModel([20, 3], ['relu', 'relu'])
model.TrainModel(learning_rate=0.3, num_iterations=30000, print_cost=True, print_iterations=10000)

model.InitializeModel([20, 3], ['relu', 'relu'], lambd=0.7)
model.TrainModel(learning_rate=0.3, num_iterations=30000, print_cost=True, print_iterations=10000)

model.InitializeModel([20, 3], ['relu', 'relu'], keep_prob=0.86)
model.TrainModel(learning_rate=0.3, num_iterations=30000, print_cost=True, print_iterations=100)