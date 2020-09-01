use std::f64::consts::PI;
use std::num;
use std::ops::{Add, Index, Mul, Sub};

const WIDTH: i64 = 900;
const HEIGHT: i64 = 900;
const INF: f64 = 1e9;
const EPS: f64 = 1e-6;

//std::mt19937 mersenneTwister;
//std::uniform_real_distribution<double> uniform;

//#define RND (2.0*uniform(mersenneTwister)-1.0)
//#define RND2 (uniform(mersenneTwister))

//typedef unordered_map<string, double> pl;

#[derive(Copy, Clone, Debug)]
struct Vector {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector {
    fn new(x: f64, y: f64, z: f64) -> Vector {
        Vector {
            x,
            y,
            z,
        }
    }

    fn scale(&self, b: f64) -> Vector {
        Vector {
            x: self.x * b,
            y: self.y * b,
            z: self.z * b,
        }
    }

    fn div(&self, b: f64) -> Vector {
        Vector {
            x: self.x / b,
            y: self.y / b,
            z: self.z / b,
        }
    }

    fn length(&self) -> f64 {
        f64::sqrt(self.dot(self))
    }

    fn norm(&self) -> Vector {
        self.div(self.length())
    }

    fn dot(&self, rhs: &Vector) -> f64 {
        let v = self * rhs;
        v.x + v.y + v.z
    }

    // Vec operator%(const Vec &b) const {return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
    fn cross(&self, rhs: &Vector) -> Vector {
        Vector {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vector {
    type Output = Vector;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul for Vector {
    type Output = Vector;

    fn mul(self, rhs: Self) -> Self::Output {
        Vector {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl<T: num::Integer> Index<T> for Vector {
    type Output = f64;

    fn index(&self, index: T) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("invalid index {}", index)
        }
    }
}


// given v1, set v2 and v3 so they form an orthonormal system
// (we assume v1 is already normalized)
fn ons(v1: &Vector) -> (Vector, Vector) {
    let v2 = if v1.x.abs() > v1.y.abs() {
        // project to the y = 0 plane and construct a normalized orthogonal vector in this plane
        let invLen = 1.0 / f64::sqrt(v1.x * v1.x + v1.z * v1.z);
        Vector::new(-v1.z * invLen, 0.0, v1.x * invLen)
    } else {
        // project to the x = 0 plane and construct a normalized orthogonal vector in this plane
        let invLen = 1.0 / f64::sqrt(v1.y * v1.y + v1.z * v1.z);
        Vector::new(0.0, v1.z * invLen, -v1.y * invLen)
    };
    let v3 = v1.cross(&v2);
    (v2, v3)
}

// Rays have origin and direction.
// The direction vector should always be normalized.
struct Ray {
    o: Vector,
    d: Vector,
}

impl Ray {
    fn new(o: Vector, d: Vector) -> Ray {
        Ray {
            o,
            d: d.norm(),
        }
    }
}

// Objects have color, emission, type (diffuse, specular, refractive)
// All object should be intersectable and should be able to compute their surface normals.
struct Obj {
    cl: Vector,
    emission: f64,
    obj_type: ObjType,
    shape: Shape,
    // void setMat(Vec cl_ = 0, double emission_ = 0, int type_ = 0) { cl=cl_; emission=emission_; type=type_; }
}

enum ObjType {
    DiffuseBRDF,
    SpecularBRDF,
    GlassRefractiveBRDF,
}

trait Intersect {
    fn intersect(&self, ray: &Ray) -> f64;
    fn normal(&self, point: &Vector) -> Vector;
}

enum Shape {
    Plane(Plane),
    Sphere(Sphere),
}

impl Intersect for Shape {
    fn intersect(self, ray: &Ray) -> f64 {
        match self {
            Plane(plane) => plane.intersect(ray),
            Sphere(sphere) => sphere.intersect(ray)
        }
    }

    fn normal(self, point: &Vector) -> Vector {
        match self {
            Plane(plane) => plane.normal(point),
            Sphere(sphere) => sphere.normal(point)
        }
    }
}

struct Plane {
    n: Vector,
    d: f64,
}

impl Intersect for Plane {
    fn intersect(self, ray: &Ray) -> f64 {
        let d0 = self.n.dot(&ray.d);
        if d0 != 0.0 {
            let t = -1 * (((n.dot(ray.o)) + d) / d0);
            if t > eps { t } else { 0.0 }
        } else {
            0.0
        }
    }

    fn normal(self, _point: &Vector) -> Vector {
        self.n
    }
}

struct Sphere {
    c: Vector,
    r: f64,
}

impl Intersect for Sphere {
    fn intersect(self, ray: &Ray) -> f64 {
        let b = ((ray.o - c).scale(2.0)).dot(&ray.d);
        let c_ = (ray.o - c).dot((&ray.o - c)) - (r * r);
        let mut disc = b * b - 4 * c_;

        if disc < 0 {
            0.0
        } else {
            disc = sqrt(disc);
            let sol1 = -b + disc;
            let sol2 = -b - disc;
            if (sol2 > eps) { sol2 / 2 } else { if (sol1 > eps) { sol1 / 2 } else { 0.0 } };
        }
    }

    fn normal(self, point: &Vector) -> Vector {
        return (point - c).norm();
    }
}

struct Intersection<'a> {
    t: f64,
    object: Option<&'a Obj>,
}

struct Scene {
    objects: Vec<Obj>
}

impl Scene {
    fn add(&mut self, o: Obj) {
        self.objects.push(o)
    }

    fn intersect(&self, ray: &Ray) -> Intersection {
        let mut closest = Intersection {
            t: INF,
            object: None,
        };
        for object in self.objects {
            let t = object.shape.intersect(ray);
            if t > eps && t < closest.t {
                closest.t = t;
                closest.object = Some(&object)
            }
        }
        closest
    }
}

// Class for generating the Halton low-discrepancy series for Quasi
// Monte Carlo integration.
struct Halton {
    value: f64,
    inv_base: f64,
}

impl Halton {
    fn number(&mut self, mut i: i32, base: i32) {
        let mut f = self.inv_base = 1.0 / base;
        self.value = 0.0;
        while i > 0 {
            self.value += f * (i % base);
            i /= base;
            f *= inv_base;
        }
    }
    fn next(&mut self) {
        let r = 1.0 - self.value - 0.0000001;
        if self.inv_base < r {
            self.value += self.inv_base;
        } else {
            let h = inv_base;
            let mut hh;
            loop {
                hh = h;
                h *= inv_base;
                if h < r {
                    break;
                }
            }
            self.value += hh + h - 1.0;
        }
    }

    fn get(&self) -> f64 {
        self.value
    }
}


// Input is the pixel offset, output is the appropriate coordinate
// on the image plane
fn camcr(x: f64, y: f64) -> Vector {
    let w: f64 = width;
    let h: f64 = height;
    let fovx = std::f64::consts::FRAC_PI_4;
    let fovy = (h / w) * fovx;
    return Vec(((2 * x - w) / w) * fovx.tan(),
               -((2 * y - h) / h) * fovy.tan(),
               -1.0);
}

// Uniform sampling on a hemisphere to produce outgoing ray directions.
// courtesy of http://www.rorydriscoll.com/2009/01/07/better-sampling/
fn hemisphere(u1: f64, u2: f64) -> Vector {
    let r = (1.0 - u1 * u1).sqrt();
    let phi = 2.0 * PI * u2;
    return Vector::new(phi.cos() * r, phi.sin() * r, u1);
}

struct Params {
    refr_index: f64
}

fn trace(ray: &mut Ray, scene: &Scene, depth: i32, clr: &mut Vector, params: &Params, hal: &Halton, hal2: &Halton) {
    // Russian roulette: starting at depth 5, each recursive step will stop with a probability of 0.1
    let mut rrFactor = 1.0;
    if (depth >= 5) {
        let rrStopProbability = 0.1;
        if RND2 <= rrStopProbability {
            return;
        }
        rrFactor = 1.0 / (1.0 - rrStopProbability);
    }

    let intersection = scene.intersect(ray);
    let (t, obj) = if let Some(obj) = intersection.object {
        (intersection.t, obj)
    } else {
        return;
    };

    // Travel the ray to the hit point where the closest object lies and compute the surface normal there.
    let hp = ray.o + ray.d.scale(t);
    let N = obj.normal(hp);
    ray.o = hp;
    // Add the emission, the L_e(x,w) part of the rendering equation, but scale it with the Russian Roulette
    // probability weight.
    *clr += Vector::new(obj.emission, obj.emission, obj.emission).scale(rrFactor);

    let mut tmp = Vector::new(0.0, 0.0, 0.0);
    match obj.obj_type {
        // Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
        ObjType::DiffuseBRDF => {
            let (rotX, rotY) = ons(N);
            let sampledDir = hemisphere(RND2, RND2);
            let rotatedDir = Vector {
                x: Vector::new(rotX.x, rotY.x, N.x).dot(&sampledDir),
                y: Vector::new(rotX.y, rotY.y, N.y).dot(&sampledDir),
                z: Vector::new(rotX.z, rotY.z, N.z).dot(&sampledDir),
            };
            ray.d = rotatedDir;    // already normalized
            let cost = ray.d.dot(N);
            trace(ray, scene, depth + 1, &mut tmp, params, hal, hal2);
            *clr = clr + (tmp.mult(obj.cl)) * cost * 0.1 * rrFactor;
        }
        // Specular BRDF - this is a singularity in the rendering equation that follows
        // delta distribution, therefore we handle this case explicitly - one incoming
        // direction -> one outgoing direction, that is, the perfect reflection direction.
        ObjType::SpecularBRDF => {
            let cost = ray.d.dot(N);
            ray.d = (ray.d - N * (cost * 2)).norm();
            trace(ray, scene, depth + 1, &mut tmp, params, hal, hal2);
            *clr = clr + tmp.scale(rrFactor);
        }
        // Glass/refractive BRDF - we use the vector version of Snell's law and Fresnel's law
        // to compute the outgoing reflection and refraction directions and probability weights.
        ObjType::GlassRefractiveBRDF => {
            let mut n = params.refr_index;
            let mut R0 = (1.0 - n) / (1.0 + n);
            R0 = R0 * R0;
            if N.dot(ray.d) > 0 { // we're inside the medium
                N = N * -1;
                n = 1 / n;
            }
            n = 1 / n;
            let cost1 = (N.dot(ray.d)) * -1; // cosine of theta_1
            let cost2 = 1.0 - n * n * (1.0 - cost1 * cost1); // cosine of theta_2
            let Rprob = R0 + (1.0 - R0) * pow(1.0 - cost1, 5.0); // Schlick-approximation
            ray.d = if cost2 > 0 && RND2 > Rprob { // refraction direction
                ((&ray.d * n) + (N * (n * cost1 - sqrt(cost2)))).norm()
            } else { // reflection direction
                (ray.d + N * (cost1 * 2)).norm()
            };
            trace(ray, scene, depth + 1, &mut tmp, params, hal, hal2);
            *clr = clr + tmp.scale(1.15 * rrFactor);
        }
    }
}

fn main() {
    unimplemented!();
}
