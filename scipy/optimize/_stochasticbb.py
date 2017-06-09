"""
stochasticbb: Stochastic branch and bound global optimization algorithm
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import copy
import scipy.optimize
import collections
from scipy._lib._util import check_random_state

__all__ = ['stochasticbb']


class Node(object):
    '''
    Class defining a node of the tree

    ndbox is an array with N elements, each a [min,max] array representing min and max values of the search space in that dimension
    eval_point is the point at which the cost function was evaluated within the ndbox
    fun is the cost fuction evaluated at eval_point
    '''

    def __init__(self, ndbox, eval_point, fun, parent):
        self.ndbox = ndbox
        self.eval_point = eval_point
        self.fun = fun
        self.children = []
        self.parent = parent

    def isLeaf(self):
        return self.children == []

    def isRoot(self):
        return self.parent is None

    def getChildren(self):
        return self.children

    def bisect(self, samplingFunc, costFunction):
        '''
        Bisect a leaf into two half-spaces along a random dimension of their ndbox. The process also updates
        the parent nodes recursively using the new sampled point and its associated cost function.
        '''

        if (self.isLeaf()):
            # Choose a random dimension to bisect the leaf
            dim = np.random.randint(0, len(self.ndbox))

            # Bisect half the space
            bisect_value = np.mean(self.ndbox[dim])

            # First node is the lower half
            ndbox1 = copy.deepcopy(self.ndbox)
            ndbox1[dim][1] = bisect_value

            eval_point = samplingFunc(ndbox1)
            fun = costFunction(eval_point)
            n1 = Node(ndbox1, eval_point, fun, self)
            n1._updateParent(fun, eval_point)

            # Second node is the upper half
            ndbox2 = copy.deepcopy(self.ndbox)
            ndbox2[dim][0] = bisect_value

            eval_point = samplingFunc(ndbox2)
            fun = costFunction(eval_point)
            n2 = Node(ndbox2, eval_point, fun, self)
            n2._updateParent(fun, eval_point)

            self.children = [n1, n2]

    def _updateParent(self, f, point):
        '''
        Recursively update parent nodes to propagate a lower cost down the tree, together with the evaluation point
        Stops either when reaching the root, or when reaching a parent node with a lower cost

        This ensures a smart traversing of the tree, as well as a simple query of the root values once the algorithm finishes
        '''
        parent_ = self.parent

        if self.isRoot():
            return True
        if (f < parent_.fun):
            parent_.fun = f
            parent_.eval_point = point
            parent_._updateParent(f, point)
        else:
            return True


class Stochastic_BB(object):
    '''
    Class defining a Tree for global optimization over a search space

    ndbox is an array with N elements, each a [min,max] array representing min and max values of the search space in that dimension
    uniformSample is a uniform sampling function over the space described by ndbox
    costFunction is a cost function 
    stopCriterion is an optional stopping criterion function. Default is simply the max iterations

    niter is the max number of iterations
    tmax and v are parameters related to the simulated annealing, where t = tmax*exp(-vi), with i the current iteration number. their default values are set for a nice decay with 100 iterations
    '''

    def __init__(self, ndbox, uniformSample, costFunction, rng, callback=None, tmax=100, tmin=0.01, niter=100):

        self.samplingFunc = uniformSample
        self.costFunc = costFunction
        self.callback = callback

        eval_point = self.samplingFunc(ndbox)
        fun = self.costFunc(eval_point)
        self.root = Node(ndbox, eval_point, fun, None)

        self.rng = rng
        self.tmax = tmax
        self.niter = niter
        self.v = (-1.0 / niter) * np.log(tmin / tmax)
        self.current_iter = 0

        self.res = scipy.optimize.OptimizeResult()

    def _computeT(self):
        '''
        Compute temperature for the annealing algorithm using the current iteration number
        '''
        return self.tmax * np.exp(-self.v * self.current_iter)

    def _selectLeaf(self):
        '''
        Select a leaf based on the evaluated function values 
        '''
        current = self.root

        while (True):
            if current.isLeaf():
                return current

            [n0, n1] = current.getChildren()

            fs0 = n0.fun
            fs1 = n1.fun

            t = self._computeT()
            p0 = t / (1 + 2.0 * t)
            p1 = (t + 1) / (1 + 2.0 * t)

            if fs0 < fs1:
                p1 = t / (1 + 2.0 * t)
                p0 = (t + 1) / (1 + 2.0 * t)

            # Select
            if self.rng.random_sample() < p0:
                current = n0
            else:
                current = n1

    def runOptimize(self, verbose=False):
        '''
        The optimization routine in itself
        '''
        self.current_iter = 0

        while (self.current_iter < self.niter):

            leaf = self._selectLeaf()
            leaf.bisect(self.samplingFunc, self.costFunc)

            self.res.x = self.root.eval_point
            self.res.fun = self.root.fun
            self.res.success = False

            if (verbose):
                print ("Iteration nr %d" % int(self.current_iter + 1))
                print ("Current f is %f " % self.res.fun)
                print ("Current x is : ")
                print (self.res.x)

            self.current_iter += 1

            # Call the callback if defined by the user
            if (self.callback is not None):
                if self.callback():
                    self.res.success = True
                    self.res.message = print(
                        "Termination criterion as defined in the callback function stopped the algorithm at iteration %d" % (self.current_iter))
                    self.res.nit = self.current_iter
                    break

        if (verbose):
            print ("Optimization successful after " +
                   str(self.current_iter) + " iterations")
            print ("Final value of f is %f " % self.res.fun)
            print ("Final x is : ")
            print (self.res.x)

        self.res.success = True
        self.res.message = "Algorithm terminated successfully after " + \
            str(self.current_iter) + " iterations"
        self.res.nit = self.current_iter
        return self.res


def uniform_sampling(ndbox):
    '''
    Uniformly sample a point inside an n-dimensional bounding box

    Parameters
----------
ndbox : ndarray
    N-dimensional bounding box, array of arrays

    Returns
    -------
    point : ndarray
            point sampled in the box		
    '''
    # Todo here, check dimensions and type of ndbox
    b = np.asarray(ndbox)
    return np.random.random_sample(b[:, 0].size) * (b[:, 1] - b[:, 0]) + b[:, 0]


def stochasticbb(func, bounds, niter=100, Tmax=100, Tmin=0.01,
                 sampling_function=None, callback=None, seed=None, disp=False):
    """
Find the global minimum of a function using a stochastic branch-and-bound algorithm [1]

Parameters
----------
func : callable ``f(x, *args)``
    Function to be optimized.  ``args`` can be passed as an optional item
    in the dict ``minimizer_kwargs``
bounds : ndarray
    Array of arrays containing the bounds of the variables to be optimized
niter : integer, optional
    The number of iterations
Tmax : float, optional
    The max "temperature" parameter for the accept or reject criterion during 
            the traversal of the tree. Higher Tmax values mean that worse solutions
            can be accepted in the first iterations, leading to a larger exploration
            of the optimization space
    Tmin : float, optional
    Tmin is the temperature at the end of the ``niter`` iterations. See [1]
            for details about Tmin and Tmax
    sampling_function : callable ``sampling_function(bounds)``, optional
            Function for uniformly sampling data points between given bounds (ndarray). Default will
            uniformly sample
    callback: callable ``callback(x, f)``, optional
            A callback function which will be called for all minima found.  ``x``
    and ``f`` are the coordinates and function value of the current minimum.
            This can be used, for example, to save the lowest N minima found.  Also,
    ``callback`` can be used to specify a user defined stop criterion by
    optionally returning True to stop the ``stochasticbb`` routine.
    seed : int or `np.random.RandomState`, optional
    If `seed` is not specified the `np.RandomState` singleton is used.
    If `seed` is an int, a new `np.random.RandomState` instance is used,
    seeded with seed.
    If `seed` is already a `np.random.RandomState instance`, then that
    `np.random.RandomState` instance is used. 
            Specify `seed` for repeatable minimizations.
    disp : bool, optional
    display verbose messages during the iterations of the algorithm



    Returns
-------
res : OptimizeResult
    The optimization result represented as a ``OptimizeResult`` object.  Important
    attributes are: ``x`` the solution array, ``fun`` the value of the
    function at the solution, and ``message`` which describes the cause of
    the termination. 
    See `OptimizeResult` for a description of other attributes.


Notes
-----
This stochastic branch-and-bound algorithm is an improvement of the classical branch-and-bound algorithm, 
    described in [1,2]. It stochastically traverses a search tree to find the global optimum of a cost function.
    Note that, as it is a stochastic algorithm, finding the minimum is only asysmptotically guaranteed with an
    infinite number of iterations.  

    Examples for using this algorithm in registration of point clouds (i.e. global optimization	over the
    SE3 group) can be found in [1,2].

    The algorithm is iterative, and each cycle is composed as such: 

    1) Stochastically traverse the tree to select a promising leaf

2) Generate children nodes to that leaf by bisecting the search space over one of its dimensions (at random)

3) For each child node, sample a point in the search space and evaluate the cost function

    4) Propagate lower values of the cost function towards the root


    The traversal of the tree (1) is governed by a temperature parameter, much like simulated annealing algorithms. 
    At each iteration i, T is computed as T = Tmax*np.exp(-v*i), where v is the cooling speed, which is computed
    from Tmax, Tmin and the max number of iterations niter. For each node of the tree, starting at the root, the
    probabiliy of selecting one of the child nodes is governed by the value of their cost functions and the
    temperature, with higher temperature yielding higher probabilities of selecting a node with a higher cost
    (see [1] for details). This process is iterated until a leaf is reached. Starting with a high temperature
    value (default Tmax is 100) ensures that a large portion of the search space is explored, while ending with a low
    temperature (default Tmin is 0.001) ensures that the algorithm converges more and more towards promising leaves
    as the iterative process goes towards its end. 

    Once a leaf is reached, children nodes are generated (2). One random dimension of the search space is selected
    and split into two halves. The new search spaces are appointed to the children. 

    For each child node, a random point is sampled in its search space, and the cost function is evaluated
    at this point (3). Values are saved as a property of the node. Note that the uniform sampling for rotations
    proposed in the original paper has been shown to be incorrect. Refer to [2] for a better sampling strategy
    in the case of rotations. 

    Finally, costs are propagated down the tree, until reaching a node with a lower cost function value, or the root. 
    This ensure a smart traversal of the tree in the next iteration, giving more resolution (splitting leaves) in 
    parts of the search space where the cost function is already low. 


    References
----------
            [1] C. Papazov and D. Burschka, Stochastic global optimization for robust point set registration,
              Comput. Vis. Image Understand. 115 (12) (2011) 1598-1609
            [2] C. Gruijthuijsen, B. Rosa, P.T. Tran, J. Vander Sloten, E. Vander Poorten, D. Reynaerts, An automatic
              registration method for radiation-free catheter navigation guidance, J. Med. Robot Res. 1 (03) (2016), 1640009
    """

    rng = check_random_state(seed)

    # Verify that the specified bounds have the correct input type and shape
    try:
        bounds_ = np.asarray(bounds, dtype=np.float32)
    except:
        raise ValueError("Wrong input data in the bounds")

    if bounds_.shape[0] < 1 or bounds_.shape[1] is not 2:
        raise ValueError(
            "Bounds must be an array of 2-element arrays, e.g. [ [1.0,2.0],[1.0,3.0], ...]")

    # Verify that there is no empy range in the specified bounds
    if np.sum(bounds_[:, 0] == bounds_[:, 1]):
        raise ValueError("Some bounds values do not specify a range")

    # sort the values so that in each dimension they are ordered as min,max
    bounds_.sort(axis=1)

    # Verify that the cost function is not set to None
    if func is not None:
        if not isinstance(func, collections.Callable):
            raise TypeError("func must be callable")
        else:
            func_wrapped = func
    else:
        raise RuntimeError("func was not set")

    # Wrap the sampling function
    if sampling_function is not None:
        if not isinstance(sampling_function, collections.Callable):
            raise TypeError("func must be callable")
        else:
            sampling_function_wrapped = sampling_function
    else:
        # If no function was defined, use basic uniform sampling
        sampling_function_wrapped = uniform_sampling

    # Verify that the cost function is callable if not set to None
    if callback is not None:
        if not isinstance(callback, collections.Callable):
            raise TypeError("func must be callable")

    optimizer = Stochastic_BB(
        bounds_, sampling_function_wrapped, func_wrapped, rng, callback, Tmax, Tmin, niter)
    return optimizer.runOptimize(verbose=disp)


### Test function from the basinhopping example ###

def _test_func2d_nograd(x):
    f = (np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
         + 1.010876184442655)
    return f


if __name__ == "__main__":
    print("Minimize a 2d function without gradient")

    bounds = [[-1, 1], [-1, 1]]

    ret = stochasticbb(_test_func2d_nograd, bounds, niter=1000,
                       Tmax=1000, Tmin=0.0001, disp=False)

    print("minimum expected at  func([-0.195, -0.1]) = 0.0")
    print(ret)
