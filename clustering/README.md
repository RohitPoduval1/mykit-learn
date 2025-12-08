# Goal
Represent a dataset by grouping its data points into a specific number of clusters, denoted as $K$

# Math
## Notation
* $K$ - The total number of clusters we want (hyperparameter)
* $\mu_k$ - The $k$th centroid, where a *centroid* is the central point that summarizes a cluster
* $r_{n}$ - The cluster the $n$th data point belongs to
	* $r_n$ is a VECTOR containing $K$ elements $r_n = \{r_{n1}, r_{n2}, \dots, r_{nk}\}$
	* A data point can only be assigned to <span style="color: hotpink;">one cluster. This is called hard assignment</span>
		* If data point $3$ was assigned to the 2nd cluster out of 4, then $r_{3} = \{0, 1, 0, 0\}$
	* <span style="color: hotpink;">Hard assignment means either a data point is in or out. Yes or no. There is NO in between. No wiggle room.</span>

## Objective Function
* Also called the "Within-Cluster Sum of Squares" (WCSS)
* Used to evaluate the compactness of the clusters
	* Lower WCSS means the clusters are tighter (points are closer to their centroid)
	* Higher WCSS means the clusters are looser (points are further from their centroid)

$$\text{WCSS} = \sum_{n=1}^N \sum_{k=1}^K r_{nk} ||x_n - \mu_k||^2$$
* $||x_n - \mu_k||^2$ - The **distance** between the $n$th data point and the $k$th cluster
	* Why L2 norm? $\rightarrow$ It makes derivations easier!
	* <span style="color: orange;">How far away is this data point away from the cluster?</span>
* $r_{nk}$ - The cluster the $n$th data point belongs to
	* Remember that $r_n$ is a vector. We only care about the data point that is actually clustered at the $\mu_k$ cluster
	* Out of all $r_{11}||x_1 - \mu_1||^2 + r_{12}||x_1 - \mu_2||^2 + ... + r_{1k}||x_1 - \mu_k||^2$, only 1 of $r_{11}, r_{12}, ..., r_{1k}$ is $1$ and that is the only one that matters. The rest of the terms are $0$
	* <span style="color: orange;">Take each data point and its respective cluster. We don’t care about the distance of a data point to the center of the other clusters</span>
* $\sum_{n=1}^N \sum_{k=1}^K$ - Sum over all data points
	* For all clusters, we want the distance between the centroids and the clusters to be as small as possible because this means we get tighter clusters (more meaningful groups)!

## Process

1. ASSIGN data points to closest prototype
2. UPDATE prototypes to be the cluster means
	* Recompute each cluster centroid to be the mean of all data points assigned to it

* 1. Minimize $J$ w.r.t. $r_{nk}$, keep $\mu_k$ fixed
	* Find the cluster the $x_n$th data point belongs to such that the distance from the data point to the cluster is minimized
	* Picks a SINGLE distance, since only 1 can be the smallest
	* Slower step than UPDATE
* 2. Minimize $J$ w.r.t. $\mu_k$, keep $r_{nk}$ fixed
	* Now that the data point has been assigned to a cluster, recompute the center of the cluster. This is computed using the following:
$$\mu_k = \frac{\sum_n r_{nk} \mathbf{x}_n}{\sum_n r_{nk}}$$
	* $\sum_n r_{nk}$
	    - The number of data points in the $k$th cluster
	    - Think `np.sum(bool_array)`

<span style="color: orange;">Note that fixing one parameter and optimizing the other finds the LOCAL optimum, not global optimum.</span>
<span style="color: orange;">All possible combinations of parameters are not tried, and hence its not global optimum</span>
