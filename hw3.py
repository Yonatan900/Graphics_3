from helper_classes import *
import matplotlib.pyplot as plt


def trace_ray(ray: Ray, ambient, lights: list[LightSource], objects: list[Object3D], depth, max_depth):
    color = np.zeros(3, dtype=np.float64)

    if depth >= max_depth:
        return color
    nearest_object, intersection_distance = ray.nearest_intersected_object(objects)
    intersection_point = ray.origin + intersection_distance * ray.direction
    # no intersection
    if nearest_object is None:
        return color
    intersection_point += nearest_object.get_normal(intersection_point) * 1e-5

    # #ambient per intersaction
    for light in lights:
        light_ray = light.get_light_ray(intersection_point)

        distance_to_light_from_intersection = light.get_distance_from_light(intersection_point)
        # shadow ray
        blocker, distance = light_ray.first_intersected_object(objects, distance_to_light_from_intersection)
        is_blocked = blocker is not None and blocker != nearest_object

        light_vector = normalize(light_ray.direction)
        light_intensity = light.get_intensity(intersection_point)
        color += nearest_object.diffuse_spec(nearest_object.get_normal(intersection_point), light_vector, ray.direction,
                                             light_intensity, is_blocked)

    reflect_ray = reflected(ray.direction, nearest_object.get_normal(intersection_point))
    color += nearest_object.reflection * trace_ray(Ray(intersection_point, reflect_ray), ambient, lights, objects,
                                                   depth + 1, max_depth)

    color += nearest_object.ambient * ambient
    return color


def render_scene(camera, ambient, lights, objects: list[Object3D], screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom
    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            viewer_direction = normalize(pixel - origin)
            camera_ray = Ray(origin, viewer_direction)
            image[i, j] = trace_ray(camera_ray, ambient, lights, objects, 1, max_depth)

            # TODO compute color
    image = np.clip(image, 0, 1)
    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    return camera, lights, objects
