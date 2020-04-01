#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "geometry.h"
#include <string>
using namespace std;
int x = 0;
int y = 0;
string filename;

struct Light {
    Light(const Vec3f& p, const float i) : position(p), intensity(i) {}
    Vec3f position;
    float intensity;
};

struct Material {
    Material(const float r, const Vec4f& a, const Vec3f& color, const float spec) : refractive_index(r), albedo(a), diffuse_color(color), specular_exponent(spec) {}
    Material() : refractive_index(1), albedo(1, 0, 0, 0), diffuse_color(), specular_exponent() {}
    float refractive_index;
    Vec4f albedo;
    Vec3f diffuse_color;
    float specular_exponent;
};

struct Sphere {
    Vec3f center;
    float radius;
    Material material;

    Sphere(const Vec3f& c, const float r, const Material& m) : center(c), radius(r), material(m) {}

    bool ray_intersect(const Vec3f& orig, const Vec3f& dir, float& t0) const {
        Vec3f L = center - orig;
        float tca = L * dir;
        float d2 = L * L - tca * tca;
        if (d2 > radius* radius) return false;
        float thc = sqrtf(radius * radius - d2);
        t0 = tca - thc;
        float t1 = tca + thc;
        if (t0 < 0) t0 = t1;
        if (t0 < 0) return false;
        return true;
    }
};

Vec3f reflect(const Vec3f& I, const Vec3f& N) {
    return I - N * 2.f * (I * N);
}

Vec3f refract(const Vec3f& I, const Vec3f& N, const float eta_t, const float eta_i = 1.f) { // Snell's law
    float cosi = -std::max(-1.f, std::min(1.f, I * N));
    if (cosi < 0) return refract(I, -N, eta_i, eta_t); // if the ray comes from the inside the object, swap the air and the media
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? Vec3f(1, 0, 0) : I * eta + N * (eta * cosi - sqrtf(k)); // k<0 = total reflection, no ray to refract. I refract it anyways, this has no physical meaning
}

bool scene_intersect(const Vec3f& orig, const Vec3f& dir, const std::vector<Sphere>& spheres, Vec3f& hit, Vec3f& N, Material& material) {
    float spheres_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < spheres.size(); i++) {
        float dist_i;
        if (spheres[i].ray_intersect(orig, dir, dist_i) && dist_i < spheres_dist) {
            spheres_dist = dist_i;
            hit = orig + dir * dist_i;
            N = (hit - spheres[i].center).normalize();
            material = spheres[i].material;
        }
    }
    
    return spheres_dist < 1000;
}

Vec3f cast_ray(const Vec3f& orig, const Vec3f& dir, const std::vector<Sphere>& spheres, const std::vector<Light>& lights, size_t depth = 0) {
    Vec3f point, N;
    Material material;

    if (depth > 4 || !scene_intersect(orig, dir, spheres, point, N, material)) {
        return Vec3f(0.2, 0.7, 0.8); // background color
    }

    Vec3f reflect_dir = reflect(dir, N).normalize();
    Vec3f refract_dir = refract(dir, N, material.refractive_index).normalize();
    Vec3f reflect_orig = reflect_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // offset the original point to avoid occlusion by the object itself
    Vec3f refract_orig = refract_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3;
    Vec3f reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, depth + 1);
    Vec3f refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;
    for (size_t i = 0; i < lights.size(); i++) {
        Vec3f light_dir = (lights[i].position - point).normalize();
        float light_distance = (lights[i].position - point).norm();

        Vec3f shadow_orig = light_dir * N < 0 ? point - N * 1e-3 : point + N * 1e-3; // checking if the point lies in the shadow of the lights[i]
        Vec3f shadow_pt, shadow_N;
        Material tmpmaterial;
        if (scene_intersect(shadow_orig, light_dir, spheres, shadow_pt, shadow_N, tmpmaterial) && (shadow_pt - shadow_orig).norm() < light_distance)
            continue;

        diffuse_light_intensity += lights[i].intensity * std::max(0.f, light_dir * N);
        specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, N) * dir), material.specular_exponent) * lights[i].intensity;
    }
    return material.diffuse_color * diffuse_light_intensity * material.albedo[0] + Vec3f(1., 1., 1.) * specular_light_intensity * material.albedo[1] + reflect_color * material.albedo[2] + refract_color * material.albedo[3];
}

void render(const std::vector<Sphere>& spheres, const std::vector<Light>& lights) {
    const int   width = x;
    const int   height = y;
    const float fov = M_PI / 3.;
    std::vector<Vec3f> framebuffer(width * height);

#pragma omp parallel for
    for (size_t j = 0; j < height; j++) { // actual rendering loop
        for (size_t i = 0; i < width; i++) {
            float dir_x = (i + 0.5) - width / 2.;
            float dir_y = -(j + 0.5) + height / 2.;    // this flips the image at the same time
            float dir_z = -height / (2. * tan(fov / 2.));
            framebuffer[i + j * width] = cast_ray(Vec3f(0, 0, 0), Vec3f(dir_x, dir_y, dir_z).normalize(), spheres, lights);
        }
    }

    std::ofstream ofs; // save the framebuffer to file
    ofs.open("./" + filename, std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height * width; ++i) {
        Vec3f& c = framebuffer[i];
        float max = std::max(c[0], std::max(c[1], c[2]));
        if (max > 1) c = c * (1. / max);
        for (size_t j = 0; j < 3; j++) {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
    ofs.close();
}

int main(int argc, char** argv) {

    Vec3f camera;
    Vec3f at;
    Vec3f up;
    int fovy;
    int lightNumber = 0;
    vector<string> lightss;
    int pigmentNumber = 0;
    vector<string> pigments;
    int surfaceNumber = 0;
    vector<string> surfaces;
    int objectNumber = 0;
    vector<string> objects;

    ifstream file("test1.in");
    if (file.is_open()) {
        string line;
        if (getline(file, line)) {
            filename = line.c_str();
        }
        if (getline(file, line)) {
            string size = line.c_str();
            string xx = size.substr(0, size.find(" "));
            string yy = size.substr(size.find(" "));
            x = stoi(xx);
            y = stoi(yy);
        }
        if (getline(file, line)) {
            string cam = line.c_str();
        }
        if (getline(file, line)) {
            string at = line.c_str();
        }
        if (getline(file, line)) {
            string up = line.c_str();
        }
        if (getline(file, line)) {
            fovy = stoi(line.c_str());
        }
        if (getline(file, line)) {
            lightNumber = stoi(line.c_str());
        }
        for (int i = 0; i < lightNumber; i++) {
            if (getline(file, line)) {
                lightss.push_back(line.c_str());
            }
        }
        if (getline(file, line)) {
            pigmentNumber = stoi(line.c_str());
        }
        for (int i = 0; i < pigmentNumber; i++) {
            if (getline(file, line)) {
                pigments.push_back(line.c_str());
            }
        }
        if (getline(file, line)) {
            surfaceNumber = stoi(line.c_str());
        }
        for (int i = 0; i < surfaceNumber; i++) {
            if (getline(file, line)) {
                surfaces.push_back(line.c_str());
            }
        }
        if (getline(file, line)) {
            objectNumber = stoi(line.c_str());
        }
        for (int i = 0; i < objectNumber; i++) {
            if (getline(file, line)) {
                objects.push_back(line.c_str());
            }
        }
        file.close();
    }

    std::vector<Light>  lights;

    for (int i = 0; i < lightNumber; i++) {
        string lightNow = lightss[i];
        int uzunluk = lightNow.substr(0, lightNow.find(" ")).length();
        int x = stoi(lightNow.substr(0, lightNow.find(" ")));
        lightNow = lightNow.substr(uzunluk);
        while (lightNow.at(0) == ' ') {
            lightNow = lightNow.substr(1);
        }

        uzunluk = lightNow.substr(0, lightNow.find(" ")).length();
        int y = stoi(lightNow.substr(0, lightNow.find(" ")));
        lightNow = lightNow.substr(uzunluk);
        while (lightNow.at(0) == ' ') {
            lightNow = lightNow.substr(1);
        }

        uzunluk = lightNow.substr(0, lightNow.find(" ")).length();
        int z = stoi(lightNow.substr(0, lightNow.find(" ")));
        lightNow = lightNow.substr(uzunluk);
        while (lightNow.at(0) == ' ') {
            lightNow = lightNow.substr(1);
        }
        cout << x << " " << y << " " << z << "\n";
        lights.push_back(Light(Vec3f(x, y, z), 1));
    }


    for (int i = 0; i < objectNumber; i++) {
        string obj = objects[i];
        
        int pigment = stoi(obj.substr(0, obj.find(" ")));
        obj = obj.substr(obj.find(" ") + 1);
        int surface = stoi(obj.substr(0, obj.find(" ")));
        obj = obj.substr(obj.find(" ")+8);
        while (obj.at(0) == ' ') {
            obj = obj.substr(1);
        }
        int uznlk = obj.substr(0, obj.find(" ")).length();
        int o1 = stoi(obj.substr(0, obj.find(" ")));
        obj = obj.substr(uznlk);
        while (obj.at(0) == ' ') {
            obj = obj.substr(1);
        }
        uznlk = obj.substr(0, obj.find(" ")).length();
        int o2 = stoi(obj.substr(0, obj.find(" ")));
        obj = obj.substr(uznlk);
        while (obj.at(0) == ' ') {
            obj = obj.substr(1);
        }
        uznlk = obj.substr(0, obj.find(" ")).length();
        int o3 = stoi(obj.substr(0, obj.find(" ")));
        obj = obj.substr(uznlk);
        while (obj.at(0) == ' ') {
            obj = obj.substr(1);
        }
        int o4 = stoi(obj.substr(0, obj.find(" ")));

        string pig = pigments[pigment];
        string sur = surfaces[surface];

        pig = pig.substr(8);
        while (pig.at(0) == ' ') {
            pig = pig.substr(1);
        }
        int r = stoi(pig.substr(0, 1));
        pig = pig.substr(1);
        while (pig.at(0) == ' ') {
            pig = pig.substr(1);
        }
        int g = stoi(pig.substr(0, pig.find(" ")));
        pig = pig.substr(1);
        while (pig.at(0) == ' ') {
            pig = pig.substr(1);
        }
        int b = stoi(pig.substr(0, pig.find(" ")));
        pig = pig.substr(1);

        // o1, o2, o3, o4
        // r, g, b
        // s1, s2, s3, s4

    }
    
    //Material      ivory(1.0, Vec4f(0.6, 0.3, 0.1, 0.0), Vec3f(0.4, 0.4, 0.3), 50.);
    //Material      glass(1.5, Vec4f(0.0, 0.5, 0.1, 0.8), Vec3f(0.6, 0.7, 0.8), 125.);
    //Material red_rubber(1.0, Vec4f(0.4, 0.6, 0.0, 1), Vec3f(1, 0, 0), 10.);
    //Material red_rubber(1.0, Vec4f(0.4, 0.6, 0.0, 0.0), Vec3f(1, 0, 0), 10.);
    //Material     mirror(1.0, Vec4f(0.0, 10.0, 0.8, 0.0), Vec3f(1.0, 1.0, 1.0), 1425.);
    /*
    Material material1(1, Vec4f(0.4, 0.6, 0.0, 0), Vec3f(0, 1, 0), 1.);
    Material material2(1, Vec4f(0.4, 0.6, 0.0, 0), Vec3f(1, 0, 0), 500.);
    Material material3(1, Vec4f(0.4, 0.6, 0.0, 0), Vec3f(0, 0, 1), 500.);

    std::vector<Sphere> spheres;
    spheres.push_back(Sphere(Vec3f(1, 0, -8), 2, material1));
    spheres.push_back(Sphere(Vec3f(3, 5, -10), 3, material2));
    spheres.push_back(Sphere(Vec3f(10, -5, -25), 10, material3));
    spheres.push_back(Sphere(Vec3f(-10, 0, -25), 10, material3));

    

    render(spheres, lights);
    */
    return 0;
}

