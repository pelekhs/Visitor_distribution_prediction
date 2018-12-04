from jsonFENCES_an import heat_lons, heat_lats, lons, lats, heat_lons2017, heat_lats2017,heat_lons2018, heat_lats2018
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from defs import mapread

X = np.column_stack((heat_lons2017, heat_lats2017))
#X = X[np.random.randint(X.shape[0], size=3000), :]

#
#from sklearn.cluster import AgglomerativeClustering
#ac = AgglomerativeClustering(n_clusters=6, linkage = 'average', affinity = 'l1')
#K = ac.fit_predict(X)
#labels = ac.labels_
#
#from sklearn.cluster import KMeans
#ac = KMeans(n_clusters=6)
#K = ac.fit_predict(X)
#labels = ac.labels_
#
#from sklearn.cluster import Birch
#ac = Birch(threshold = 0.00001, n_clusters = 6)
#K = ac.fit_predict(X)


from sklearn.cluster import DBSCAN
ac = DBSCAN(eps=0.000355, min_samples=700, n_jobs = -1)
K = ac.fit_predict(X)


heats = np.column_stack((X,K))

def sort(a, number):
    heat = a[np.where(a[:,2] == number)]
    return heat
  
if __name__ == "__main__":

    heat0 = sort(heats, 0)
    heat1 = sort(heats, 1)
    heat2 = sort(heats, 2)
    heat3 = sort(heats, 3)
    heat4 = sort(heats, 4)
    heat5 = sort(heats, 5)
#    heat6 = sort(heats, 6)
#    heat7 = sort(heats, 7)
#    vor=Voronoi(centers)
#    regions, vertices = voronoi_finite_polygons_2d(vor)
#       
    fig, ax = plt.subplots()
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")

    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    plt.scatter(heat0[:,0], heat0[:,1], alpha=0.9, c=color[0], s=5, edgecolor='')
    plt.scatter(heat1[:,0], heat1[:,1], alpha=0.9, c=color[1], s=5, edgecolor='')
    plt.scatter(heat2[:,0], heat2[:,1], alpha=0.9, c=color[2], s=5, edgecolor='')
    plt.scatter(heat3[:,0], heat3[:,1], alpha=0.9, c=color[3], s=5, edgecolor='')
    plt.scatter(heat4[:,0], heat4[:,1], alpha=0.9, c=color[4], s=5, edgecolor='')
    plt.scatter(heat5[:,0], heat5[:,1], alpha=0.9, c=color[5], s=5, edgecolor='')
#    plt.scatter(heat6[:,0], heat6[:,1], alpha=0.9, c='orange', s=1, edgecolor='')
#    plt.scatter(heat7[:,0], heat7[:,1], alpha=0.9, c='grey', s=1, edgecolor='')
    
    
#    plt.scatter(poi_centers[:,0], poi_centers[:,1], s=80, c='brown')
#    
#    for point in centers:
#        plt.scatter(point[0], point[1], s=50, c='black')
#        

    #listaki = [heat0, heat1, heat2, heat3, heat4]
    img = mapread()
    i=0
    plt.imshow(img, alpha=0.5, extent=[lons[0],lons[2],lats[0], lats[2]])

#    for region in regions:
#        we = vertices[region]
#        plt.fill(*zip(*we), alpha=0.2, c=color[i], edgecolor='black')
#        i += 1
    plt.xlim(min(lons), max(lons))
    plt.ylim(min(lats), max(lats))
    
    plt.savefig('DBSCANclustering.png')
    plt.show()

