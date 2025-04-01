![vgan1-light](https://github.com/user-attachments/assets/770fe2f6-8c42-4e4d-b7bd-bf015c46f993)
============================================================================


# Introduction 
Repository for the V-GAN algorithm in the paper "Adversarial Subspace Generation for Knowledge Discovery in High Dimensional Data" [1] for subspace search.

Our proposed algorithm, V-GAN, is capable of identifying a collection of subspaces relevant to a studied population $\mathbf{x}$. We do so by building on a theoretical framework that explains the _Multiple Views_ phenomenon of data. 
These subspaces have the special property that *randomly looking at any of them, when sampling from* $\mathbf{x}$, *maintains the distribution*.

## What does this mean, and why should I care?

Dealing with tabular data is hard, mainly because finding proper patterns in structured data can be a quite challenging task, where even Deep Methods fail to extract proper patterns from the data [2].
Some authors theorized that this complexity primarily comes from the *seemingly random* aggregation of the data in certain subspaces of the data [3,4], usually referred as the *Multiple View effect*. 


Turns out that one can actually model this *subspace* behavior of the data, by considering a random projection $\mathbf{U}(\omega)=U$, where $\omega \in (\Omega,\sigma, \mathbb{P})$ and $U$ is a projection matrix. This way, if the $\mathbf{U}$ is carefully selected to verify that:  $\mathbb{P}$<sub>$\mathbf{Ux}$</sub> = $\mathbb{P}$<sub>$\mathbf{x}$</sub>, randomly looking the data in the subspaces obtained when sampling realizations of $\mathbf{U}$ is the same as looking at the data in the full-space.
Therefore, the *Multiple Views effect* occurs whenever a random projection like this exists for a given population $\mathbf{x}$. Needless to say, being able to exploit such structures in the data has the potential to reveal interesting subspaces for Dimensionality Reduction, Ensembling, and even xAI applications. In our research [1], we focused on the use of such subspaces for ensembling in the Outlier Detection problem. 


## What are these random projections, really?

In theory, the $\mathbf{U}$ do not necessarily need to be projections &mdash; although it is the most straightforward use case in practice. In our research, we focused on considering axis-parallel projections, i.e., feature subspaces of the data. We do so mainly for simplicity, as they are easier to learn and interpret. In practice, the projections tend to select the most important collection of features that characterize a given distribution. While this can be very difficult to visualize and interpret with tabular data, image data gives a clearer example of this: 


<img src=https://github.com/user-attachments/assets/033a9420-b8bd-4777-9ce5-9d599018d0e4 alt="drawing" width="500"/>

In this case, each feature corresponds to one pixel, thus, the subspaces correspond to particular parts of an image being selected. The main benefit of this random projection is that the application to tabular data is direct, thus one could expect a similar level of representation for tabular data &mdash; something that is not always guaranteed [2,3].

## Okey, but how do you get $\mathbf{U}$?

The answer is with V-GAN! V-GAN is a generative network focused on obtaining a random axis-parallel projection $\mathbf{V}$ by minimizing the squared Maximum Mean Discrepancy loss between $\mathbb{P}$<sub>$\mathbf{Ux}$</sub> and $\mathbb{P}$<sub>$\mathbf{x}$</sub>.
This way, V-GAN takes a noise input $z$ to output a linear projection $V$, represented as a binary diagonal matrix. 

<img src=https://github.com/user-attachments/assets/b18f2a2c-9f20-4104-ae24-dfb721952a68 alt="vgan" width="500"/>

This way, by randomly selecting different values for the input noise $z$, one can sample different realizations from the underlying distribution of the learned $\mathbf{V}$.

# Installation



To install, simply use the requirements.txt file 
`pip install -r requirements.txt`
Additionally, if you plan to train VGAN, you should also install the torch-two-sample package: 

```
git clone git@github.com:josipd/torch-two-sample.git
cd torch-two-sample
pip install .
```

# How to use VGAN, and how to reproduce our results

## For reproducing our results
The main branch of the repository constitutes the current release of V-GAN, and it's meant to be used on the final user's data. To reproduce our results, you can use the specific code **on each branch**. The data for the real-world experiments can also be downloaded from the original source (the ADBench benchmark [4]) by simply executing 
```
sh get_data.sh
```
in any of our experimental branches.

## For your own data

Is very easy to use V-GAN for your own data! An example can be found in the notebook `test.ipynb`. 

# References 
[1] Adversarial Subspace Generation for Knowledge Discovery in High-Dimensional Data

[2] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. 2022. Why do tree-based models still outperform deep learning on typical tabular data? In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 37, 507–520.

[3] Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2024). Deep Neural Networks and Tabular Data: A Survey. IEEE Transactions on Neural Networks and Learning Systems, 35(6), 7499–7519.

[4] Songqiao Han, Xiyang Hu, Hailiang Huang, Minqi Jiang, and Yue Zhao. 2022. ADBench: anomaly detection benchmark. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 2329, 32142–32159.
