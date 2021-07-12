# CS230Project
 Winter 2021 Project CS230

ABSTRACT

Post-earthquake damage surveys currently require teams of domain experts to
visually inspect buildings to determine their safety, which is slow and subjective.
Efforts to automate the process using computer vision have been limited due to the
time and resource cost of labeling earthquake survey images. In this project I use
pseudo-labeling to take advantage of large numbers of unlabeled reconnaissance
images available on the web in a semi-supervised learning approach. I investigate
the effects of changing the start-epoch of pseudo-labeling during training. Results
show that prediction accuracy improved by up to 5% on a test set from the unlabeled
data, although improvement is sensitive to the accuracy of the base model.

PAPER
http://cs230.stanford.edu/projects_winter_2021/reports/70759638.pdf

DATASET
https://apps.peer.berkeley.edu/phi-net/
Gao Y., & Mosalam K.M. (2018). Deep Transfer Learning for Image-based Structural Damage Recognition, Computer-Aided Civil and Infrastructure Engineering, 33(9): 748-768.
Gao, Y., & Mosalam, K. M. (2019). PEER Hub ImageNet (Φ-Net): A Large-Scale Multi-Attribute Benchmark Dataset of Structural Images. PEER Report No.2019/07, Pacific Earthquake Engineering Research Center, University of California, Berkeley, CA.
Gao, Y., & Mosalam, K. M. (2020). PEER Hub ImageNet: A Large-Scale Multiattribute Benchmark Data Set of Structural Images. Journal of Structural Engineering, 146(10), 04020198.
(Usage requires Tasks 1 and 2)

COMMAND LINE ARGUMENTS
--task
    Task 1: Scene Level (Pixel, Object, Structure), Task 2: Damage State (Damaged, Undamaged)
--semisupervised
    True/False use unlabeled images for pseudo-label training
--path
    Location of folders containing datasets for each task
--val_split
    Proportion of (labeled) training set to use as validation
--batch_size
    Number of examples per training batch
--epochs
    Number of epochs to train from data
--lr
    Adam optimizer learning rate
--alpha_range
    List of length two defining the epoch to start including pseudo-labels in loss function, and the epoch to end the ramp-up of loss weighting
--crop_dataset
    Decrease the size of the training dataset for debugging speed

USAGE EXAMPLES

Supervised:
python main.py --task 1 --epochs 20

Semisupervised:
python main.py --task 1 --semisupervised --alpha_range 3 5 --epochs 5
