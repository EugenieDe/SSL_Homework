# SSL_Homework
![image](https://github.com/EugenieDe/SSL_Homework/assets/104826934/5030641b-eddd-496c-8873-1ed436093bbe)

## Main Requirements
* torch = 1.13.0
* torchvision = 0.14.0
* tqdm = 4.64.1

## Part 1: VICReg on CIFAR-10 (2 points)
In this tutorial, you are going to implement VICReg on the CIFAR-10 dataset. \
a) Recall the principle of VICReg (hint: it is the same as the global criterion of VICRegL that we saw in class). \
b) Look at the projector (also called expander) in model.py. What is its role? \
c) Look at loss.py and train.py. Now implement the loss under the TODO in line 60.

## Part 2: Evaluation (1 point)
To evaluate the performance of your model, you are going to implement a linear classifier and calculate the accuracy of your model on the CIFAR-10 dataset. \
a) Look at eval.py. Implement the linear classifier on lines 24 and 58. \
b) As you can see, there is a training step on the CIFAR-10 dataset. What exactly are we training here?

## Part 3: Hyperparameters Experiments (1 point)
Run at least 3 experiments varying the hyperparameters of your choice. Attach a table with the results to your report.

## Part 4: Generalization (1 point)
You are going to evaluate VICReg on a classification task on another dataset to test its generalization capabilities. \
a) Look up the German Traffic Sign Recognition Benchmark. Give a brief description of it. \
b) In eval.py, evaluate the performance of your previously trained model on the GTSRB dataset (you can find it in torchvision.datasets). Discuss the results.

# Deadline and Report
Please upload to your repository a PDF file names Lastname_SSL.pdf. \
Deadline: Oct 16, 2023, 23:59
# References
VICREG: VARIANCE-INVARIANCE-COVARIANCE REGULARIZATION FOR SELF-SUPERVISED LEARNING: https://arxiv.org/pdf/2105.04906.pdf \
VICRegL: Self-Supervised Learning of Local Visual Features: https://proceedings.neurips.cc/paper_files/paper/2022/file/39cee562b91611c16ac0b100f0bc1ea1-Paper-Conference.pdf
# Credits
The code in this respository is a modified version of https://github.com/augustwester/vicreg.
