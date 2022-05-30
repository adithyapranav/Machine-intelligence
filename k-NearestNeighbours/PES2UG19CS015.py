import numpy as np
import copy
import statistics


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        m=len(self.data)
        n=len(x)
        no_of_rows=len(x[0])
        distance=[[0 for i in range(m)] for j in range(n)] #creating 0 matrix
        for i in range(n):
            for j in range(m):
                for k in range(no_of_rows):
                    distance[i][j]+=abs(x[i][k]-self.data[j][k])**self.p
                distance[i][j]=distance[i][j]**(1/self.p)
        return distance
        pass

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        n=len(x)
        no_of_neigh=self.k_neigh
        neigh_dists=[[0 for i in range(no_of_neigh)] for j in range(n)] 
        idx_of_neigh=[[0 for i in range(no_of_neigh)] for j in range(n)] 
        distance_matrix=self.find_distance(x)
        for i in range(n):
            for j in range(no_of_neigh):
                smallest_element=min(distance_matrix[i])
                smallest_element_id=distance_matrix[i].index(smallest_element)
                neigh_dists[i][j]=smallest_element
                idx_of_neigh[i][j]=smallest_element_id
                distance_matrix[i][smallest_element_id]=float('inf')	
        return (neigh_dists, idx_of_neigh)              
        pass

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        neigh_dists,idx_of_neigh=self.k_neighbours(x)
        x=np.array(x)
        shp_x=np.shape(x)
        temp=[-1 for n in range(self.k_neigh)]
        pred=[-1 for m in range(shp_x[0])]
        for i in range(shp_x[0]):
                for j in range(self.k_neigh):
                        temp[j]=self.target[idx_of_neigh[i][j]]
                        temp= np.array(temp)
                        values,counts = np.unique(temp, return_counts=True)
                        indx = np.argmax(counts)
                        pred[i]=values[indx]
        return pred
        pass

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        our_prediction=self.predict(x)
        no_of_correct_pred=0
        indx=0
        for i in our_prediction:
            if our_prediction[indx]==y[indx]:
                no_of_correct_pred+=1
            indx+=1
        return (no_of_correct_pred/len(x))*100
        pass