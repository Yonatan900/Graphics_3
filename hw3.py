from helper_classes import *
import matplotlib.pyplot as plt


# ##
# RGB GetColor(Scene scene, Ray ray, Point hit, int level) {
# // Ambient and Emission calculations
# RGB color = calcEmissionColor(scene)
# + calcAmbientColor(scene);
# // Diffuse & Specular calculations
# for (int j = 0; j < getNumLights(scene); j++) {
# Light light = getLight(j,scene);
# color = color
# + calcDiffuseColor(scene,hit,ray,light)
# + calcSpecularColor(scene,hit,ray,light);
# }
# level = level+1;
# if (level > MAX_LEVEL)
# return color;
# // reflective & refractive calculations
# Ray r_ray = ConstructReflectiveRay(ray, scene, hit);
# Point r_hit = FindIntersection(r_ray, scene)
# color += K_R * GetColor(scene, r_ray, r_hit, level);
# Ray t_ray = ConstructReflefractiveRay(ray, scene, hit);
# Point t_hit = FindIntersection(t_ray, scene)
# color += K_T * GetColor(scene, t_ray, t_hit, level);
# return color;
# }
##
def getColor(color):
    pass


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
            nearest_object, intersection_distance = camera_ray.nearest_intersected_object(objects)

            intersection_point = camera_ray.origin + intersection_distance * camera_ray.direction
            intersection_point += nearest_object.normal * 1e-5
            color = np.zeros(3)
            for light in lights:
                light_ray = light.get_light_ray(intersection_point)
                blocker, _ = light_ray.nearest_intersected_object(objects)
                if blocker is None:
                    light_vector = normalize(light_ray.direction)
                    light_intensity = light.get_intensity(intersection_point)
                    color += nearest_object.phong_color(nearest_object.normal, light_vector, camera_ray.direction,
                                                        light_intensity, ambient)

            # This is the main loop where each pixel color is computed.
            image[i, j] = color
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
