# K-Means Clustering

# Importing the libraries
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from voronoi import voronoi_finite_polygons_2d


#my Kmeans inherits the well known KMeans of sklearn
class my_kmeans(KMeans):
    def __init__(self, lon, lat):
        self.X = lon
        self.Y = lat
        self.point_vector = np.column_stack((lon,lat))
        self.y_kmeans = None
        self.wcss = []
        self.clusters = 0

    #method to compute the optimal number of clusters(graphically)
    def elbow_method_plot(self,clusters):
        if clusters > 0:
            wcss = []
            plt.figure()
            for i in range(1, clusters+1):
                kmeans = KMeans(n_clusters = i, init = 'k-means++',n_jobs=-1)
                kmeans.fit(self.point_vector)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1, clusters+1), wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('elbow_method_clustering.png')
        else:
            raise ValueError("Wrong number of clusters. Need smth above 0")

    #this method does the clustering and optionally plots on image or blank with voronoi or not
    def clustering(self, clusters = 0, plot = False, vor_split = False, img = [], x_border = [], y_border = [], filetosave = '', point_radius = 1, random_state = 42):
        if clusters > 0:
            self.clusters = clusters
            # Fitting K-Means to the dataset
            kmeans = KMeans(n_clusters = clusters , init = 'k-means++', random_state = None, n_jobs=-1)
            self.y_kmeans = kmeans.fit_predict(self.point_vector)
            self.centers = kmeans.cluster_centers_
        else:
            raise ValueError("Negative number of clusters. Enter positive cluster number")

        # Visualising the clusters:
        if (plot):
            plt.close()
            plt.figure()
            #cluster plot
            z=['orange', 'green', 'blue', 'purple', 'red', 'yellow',
               'magenta', 'black', 'cyan', 'grey', 'pink', 'beige']
            for i in range(self.clusters):
                plt.scatter(self.point_vector[self.y_kmeans == i, 0],  
                            self.point_vector[self.y_kmeans == i, 1],
                            s = point_radius, c = z[i%11],
                            label = 'Cluster' + str(chr(65+i+1)), edgecolor ='')

            #centroid plot
            plt.scatter(kmeans.cluster_centers_[:, 0],  
                        kmeans.cluster_centers_[:, 1], 
                        s = 100, c = 'yellow', edgecolor ='black',label = 'Centroids')
            plt.show()

            if vor_split:
                #voronoi plot
                vor = Voronoi(kmeans.cluster_centers_)
                regions, vertices = voronoi_finite_polygons_2d(vor)
                color=['orange', 'green', 'blue', 'purple', 'red', 'yellow',
                       'magenta', 'black', 'cyan', 'grey', 'pink', 'beige']

                #dimension fix and map plot
                i = 0
                for region in regions:
                    we = vertices[region]
                    plt.fill(*zip(*we), alpha=0.2, c=color[i % 11], edgecolor='black')
                    i += 1

            if  ((len(x_border) != 0) and (len(y_border) != 0)):
                plt.xlim(min(x_border), max(x_border))
                plt.ylim(min(y_border), max(y_border))

                if (len(img) != 0):
                    plt.imshow(img, alpha=0.5, extent=[x_border[0],x_border[2],
                                                       y_border[0], y_border[2]])
            else:
                if (len(img) != 0):
                    plt.imshow(img, alpha=0.5)

            #plot options
            plt.title('Clusters of geotags')
            plt.ylabel('Latitude')
            plt.xlabel('Longitude')
            plt.legend(markerscale=3)
            plt.show()
            if (filetosave != ''):
                plt.savefig(filetosave)
        return plt
