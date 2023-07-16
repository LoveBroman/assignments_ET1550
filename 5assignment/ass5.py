import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Uppgift 1 K-means clustering.

class Point:
   def __init__(self, x, y):
       self.x = x
       self.y = y

   def __str__(self):
       return f"Point({self.x, self.y})"

   def __repr__(self):
       return self.__str__()

   def __hash__(self):
      return hash((self.x, self.y))


   def __eq__(self, other):
       return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9


   def sq_distance(self, other):
       return  (self.x - other.x) ** 2 + (self.y -other.y) ** 2

class Centroid(Point):
    def __init__(self, x, y, points=None):
        super().__init__(x, y)
        if points is None:
            self.points = set()

    def __str__(self):
        return f"Centroid ({self.x, self.y}), Elements" + "".join([f' |{p}|\n ' for p in self.points])



    def clear(self):
        self.points = set()

    def __repr__(self):
        return self.__str__()

    def recalc_centroid(self):
        num_points = len(self.points)
        new_x, new_y =sum([p.x for p in self.points])/num_points, sum([p.y for p in self.points])/num_points
        if (self.x == new_x) and (self.y == new_y):
            return False
        else:
            self.x = new_x
            self.y = new_y
            return True

    def add_point(self, point):
        self.points.add(point)

def assign_cent(point, centroids):
    #print(f"Before{centroids} Before ")
    sorted(centroids, key=lambda p: point.sq_distance(p))[0].add_point(point)
    #print(f"After {centroids} After")

def get_centroid(points):
    return Point(sum([p.x for p in points]), sum([p.y for p in points]))

def assign_to_all(points, centroids):
    for point in points:
        assign_cent(point, centroids)

def k_means(points, centroids):
    for cent in centroids:
        cent.clear()
    assign_to_all(points, centroids)
    changed_cent = any([cent.recalc_centroid() for cent in centroids])
    if changed_cent:
        return k_means(points, centroids)
    else:
        return centroids

def cost(centroids):
    return sum([sum([p.sq_distance(c) for p in c.points]) for c in centroids])


points = [Point(2,1), Point(2.5, 2.2), Point(1.8, 1.8), Point(5,6), Point(5.5, 7), Point(4.5, 5.5)]
centroids = [Centroid(2, 1), Centroid(5, 6)]

ck = k_means(points, centroids)

x_vals0 = [p.x for p in centroids[0].points]
x_vals1 = [p.x for p in centroids[1].points]
y_vals0 = [p.y for p in centroids[0].points]
y_vals1 = [p.y for p in centroids[1].points]
plt.scatter(x_vals0, y_vals0)
plt.scatter(x_vals1, y_vals1)
plt.scatter(centroids[0].x, centroids[0].y)
plt.scatter(centroids[1].x, centroids[1].y)
plt.show()

# Fråga 2
kostnad1 = cost(ck)
print(kostnad1)

c2 = [Centroid(4.75, 5.75), Centroid(2.1, 1.67), Centroid(5.5, 7)]

assign_to_all(points, c2)
print(cost(c2))

# Fråga 3
x = np.array([0.15, -0.66, 1.58, -1.77, 0.96, -0.86])
v1 = np.array([0.35, 0.5, 0.45, 0.4, 0.2, 0.4])
v2 = np.array([-0.4, 0.2, -0.1, 0.6, -0.45, 0.45])
v3 = np.array([-0.1, -0.1, -0.3, -0.2, 0.8, -0.4])
U = np.array([v1, v2, v3])

print(np.matmul(U,x.T))

#Fråga 4

x = np.array([0.3775, 0.0511, 0.0279, 0.0230, 0.0168, 0.0120, 0.0085, 0.0039, 0.0018])

k=0
v = 1
vi = 1

while vi > 0.90:
    x_l = (x ** 2).sum()
    x_u = (x[:len(x)-k] ** 2).sum()
    v = vi
    vi = x_u / x_l
    k+=1
    print(k)
    print(v)


print("barbar",(x[:-8] ** 2).sum() / (x ** 2).sum())
#Svara enbart den första principalkomponenten behövs.

print(v)