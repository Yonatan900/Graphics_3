import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    axis = axis / np.linalg.norm(axis)
    dot_product = np.dot(vector, axis)
    reflection = vector - 2 * dot_product * axis

    return normalize(reflection)


# TODO unreachable problem
def cross(vec1, vec2):
    return np.cross(vec1, vec2)


## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity

    def get_distance_from_light(self, intersection):
        """abstract"""
        pass

    def get_intensity(self, intersection_point):
        """abstract"""
        pass

    def get_light_ray(self, intersection_point):
        """abstract"""
        pass


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)
        # TODO

    # This function returns the ray that goes from the point to a light source
    def get_light_ray(self, intersection_point):
        # TODO documentation mistake?
        return Ray(origin=intersection_point, direction=-self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        attenuation = self.kc + self.kl * d + self.kq * (d ** 2)
        return self.intensity / attenuation


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq
        # TODO

    # This function returns the ray that goes from the point to a light
    def get_light_ray(self, intersection):
        # TODO   mistake?
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)
        # TODO

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        ray = normalize(intersection - self.position)
        cos_theta = np.dot(ray, self.direction)
        cos_theta = max(cos_theta, 0)
        attenuation = self.kc + self.kl * d + self.kq * (d ** 2)

        return self.intensity * cos_theta / attenuation
        # TODO


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for object_i in objects:
            intersection = object_i.intersect(self)
            if intersection is not None:
                d, inter_object = intersection
                if d < min_distance:
                    nearest_object = inter_object
                    min_distance = d
        # TODO
        return nearest_object, min_distance

    def first_intersected_object(self, objects, max_distance=np.inf):
        for object_i in objects:
            intersection = object_i.intersect(self)
            if intersection is not None:
                d, inter_object = intersection
                if d < max_distance:
                    return inter_object, d
        return None, None


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

    # abstractmethod
    # TODO calrify abstarct
    def intersect(self, ray):
        """
        Return the distance to the intersection point and the object itself,
        or None if there is no intersection.
        """

        pass

    def get_normal(self, inter_p):
        """abstract"""
        pass

    def diffuse_spec(self, n_vec, l_vec, v_vec, i_light, is_shadowed=False):
        ref_l = reflected(l_vec, n_vec)
        cosn_theta = max(0, np.dot(v_vec, ref_l)) ** self.shininess

        diffuse = self.diffuse * i_light * max(0, np.dot(n_vec, l_vec))
        specular = self.specular * i_light * cosn_theta

        return (diffuse + specular) * int(not is_shadowed)


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def get_normal(self, inter_p):
        return self.normal

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """

    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def get_normal(self, inter_p):
        return self.normal

    # computes normal to the triangle surface.
    def compute_normal(self):
        # Vectors for two edges of the triangle
        vec1 = self.b - self.a
        vec2 = self.c - self.a
        normal = cross(vec1, vec2)

        # Normalize the resulting vector
        return normalize(normal)

    def intersect_triangle_plane(self, ray: Ray):
        v = self.a - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None

    def intersect(self, ray: Ray):
        # TODO float problem?

        res = self.intersect_triangle_plane(ray)
        if res is None:
            return None
        t, _ = res
        point_p = ray.origin + t * ray.direction
        v_AB = self.b - self.a
        v_AC = self.c - self.a
        v_PB = point_p - self.b
        v_PC = point_p - self.c
        v_PA = point_p - self.a

        area = np.linalg.norm(cross(v_AC, v_AB)) / 2
        if area == 0:
            print("Area is 0")
            return None

        alpha = np.linalg.norm(cross(v_PB, v_PC)) / (2 * area)
        beta = np.linalg.norm(cross(v_PC, v_PA)) / (2 * area)
        gamma = np.linalg.norm(cross(v_PB, v_PA)) / (2 * area)
        if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1) and (
                1 - 1e-6 <= alpha + beta + gamma <= 1 + 1e-6):
            return t, self
        return None
        # TODO


class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """

    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
            [0, 1, 3],
            [1, 2, 3],
            [0, 3, 2],
            [4, 1, 0],
            [4, 2, 1],
            [2, 4, 0]]
        for idx in t_idx:
            l.append(Triangle(self.v_list[idx[0]], self.v_list[idx[1]], self.v_list[idx[2]]))

        return l

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        tri, d = ray.nearest_intersected_object(self.triangle_list)
        return d, tri


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def get_normal(self, inter_p):
        v_norm = inter_p - self.center
        return normalize(v_norm)

    def intersect(self, ray: Ray):
        L = self.center - ray.origin
        t_c = np.dot(L, ray.direction)
        d2 = np.dot(L, L) - t_c ** 2
        r2 = self.radius ** 2

        if d2 > r2:
            return None  # No intersection

        t_h = np.sqrt(r2 - d2)
        t0 = t_c - t_h
        t1 = t_c + t_h

        # both points are behind the ray origin
        if t0 < 0 and t1 < 0:
            return None

        # We return the smallest non-negative t value, the closest intersection
        t = t0 if t0 > 0 else t1
        return t, self
