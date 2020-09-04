use rand::prelude::random;
use rayon::prelude::*;
use std::cmp::min;
use std::f64::consts::{FRAC_PI_4, PI};
use std::fs::File;
use std::io::Write;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::sync::atomic::Ordering;
use std::time::Instant;

const EPS: f64 = 1e-6;

#[derive(Copy, Clone, Debug)]
struct V3 {
    x: f64,
    y: f64,
    z: f64,
}

impl V3 {
    fn new(x: f64, y: f64, z: f64) -> V3 {
        V3 { x, y, z }
    }

    fn scale(&self, b: f64) -> V3 {
        V3 {
            x: self.x * b,
            y: self.y * b,
            z: self.z * b,
        }
    }

    fn div(&self, b: f64) -> V3 {
        V3 {
            x: self.x / b,
            y: self.y / b,
            z: self.z / b,
        }
    }

    fn length(&self) -> f64 {
        f64::sqrt(self.dot(self))
    }

    fn norm(&self) -> V3 {
        self.div(self.length())
    }

    fn dot(&self, rhs: &V3) -> f64 {
        let v = *self * *rhs;
        v.x + v.y + v.z
    }

    fn cross(&self, rhs: &V3) -> V3 {
        V3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl Add for V3 {
    type Output = V3;

    fn add(self, rhs: Self) -> Self::Output {
        V3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for V3 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other
    }
}

impl Sub for V3 {
    type Output = V3;

    fn sub(self, rhs: Self) -> Self::Output {
        V3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul for V3 {
    type Output = V3;

    fn mul(self, rhs: Self) -> Self::Output {
        V3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

// given v1, set v2 and v3 so they form an orthonormal system
// (we assume v1 is already normalized)
fn ons(v1: &V3) -> (V3, V3) {
    let v2 = if v1.x.abs() > v1.y.abs() {
        // project to the y = 0 plane and construct a normalized orthogonal vector in this plane
        let invLen = 1.0 / f64::sqrt(v1.x * v1.x + v1.z * v1.z);
        v3(-v1.z * invLen, 0.0, v1.x * invLen)
    } else {
        // project to the x = 0 plane and construct a normalized orthogonal vector in this plane
        let invLen = 1.0 / f64::sqrt(v1.y * v1.y + v1.z * v1.z);
        v3(0.0, v1.z * invLen, -v1.y * invLen)
    };
    let v3 = v1.cross(&v2);
    (v2, v3)
}

// Rays have origin and direction.
// The direction vector should always be normalized.
struct Ray {
    o: V3,
    d: V3,
}

// Objects have color, emission, type (diffuse, specular, refractive)
// All object should be intersectable and should be able to compute their surface normals.
struct Obj {
    cl: V3,
    emission: f64,
    obj_type: ObjType,
    shape: Shape,
}

enum ObjType {
    DiffuseBRDF,
    SpecularBRDF,
    GlassRefractiveBRDF,
}

trait Intersect {
    fn intersect(&self, ray: &Ray) -> f64;
    fn normal(&self, point: &V3) -> V3;
}

enum Shape {
    Plane(Plane),
    Sphere(Sphere),
}

impl Intersect for Shape {
    fn intersect(&self, ray: &Ray) -> f64 {
        match self {
            Self::Plane(plane) => plane.intersect(ray),
            Self::Sphere(sphere) => sphere.intersect(ray),
        }
    }

    fn normal(&self, point: &V3) -> V3 {
        match self {
            Self::Plane(plane) => plane.normal(point),
            Self::Sphere(sphere) => sphere.normal(point),
        }
    }
}

struct Plane {
    n: V3,
    d: f64,
}

impl Intersect for Plane {
    fn intersect(&self, ray: &Ray) -> f64 {
        let d0 = self.n.dot(&ray.d);
        if d0 != 0.0 {
            let t = -1.0 * (((&self.n.dot(&ray.o)) + &self.d) / d0);
            if t > EPS {
                t
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn normal(&self, _point: &V3) -> V3 {
        self.n
    }
}

struct Sphere {
    c: V3,
    r: f64,
}

impl Intersect for Sphere {
    fn intersect(&self, ray: &Ray) -> f64 {
        let to_o = ray.o - self.c;
        let b = (to_o.scale(2.0)).dot(&ray.d);
        let c_ = to_o.dot(&to_o) - (self.r * self.r);
        let mut disc = b * b - 4.0 * c_;

        if disc < 0.0 {
            0.0
        } else {
            disc = disc.sqrt();
            let sol1 = -b + disc;
            let sol2 = -b - disc;
            if sol2 > EPS {
                sol2 / 2.0
            } else if sol1 > EPS {
                sol1 / 2.0
            } else {
                0.0
            }
        }
    }

    fn normal(&self, point: &V3) -> V3 {
        (*point - self.c).norm()
    }
}

struct Intersection<'a> {
    t: f64,
    object: Option<&'a Obj>,
}

struct Scene {
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    refr_index: f64,
    background_color: V3,
    fovx: f64,
    objects: Vec<Obj>,
}

impl Scene {
    fn add(&mut self, o: Obj) {
        self.objects.push(o)
    }

    fn intersect(&self, ray: &Ray) -> Intersection {
        let mut closest = Intersection {
            t: std::f64::INFINITY,
            object: None,
        };
        for object in &self.objects {
            let t = object.shape.intersect(ray);
            if t > EPS && t < closest.t {
                closest.t = t;
                closest.object = Some(&object)
            }
        }
        closest
    }

    // Input is the pixel offset, output is the appropriate coordinate
    // on the image plane
    fn camcr(&self, x: f64, y: f64) -> V3 {
        let w: f64 = self.width as f64;
        let h: f64 = self.height as f64;
        let fovy = (h / w) * self.fovx;
        v3(
            ((2.0 * x - w) / w) * self.fovx.tan(), // tan pi/4 = 1
            -((2.0 * y - h) / h) * fovy.tan(),
            -1.0,
        )
    }
}

// Uniform sampling on a hemisphere to produce outgoing ray directions.
// courtesy of http://www.rorydriscoll.com/2009/01/07/better-sampling/
fn hemisphere(u1: f64, u2: f64) -> V3 {
    let r = (1.0 - u1 * u1).sqrt();
    let phi = 2.0 * PI * u2;
    v3(phi.cos() * r, phi.sin() * r, u1)
}

fn trace(ray: &mut Ray, scene: &Scene, depth: i32, clr: &mut V3) {
    // Russian roulette: starting at depth 5, each recursive step will stop with a probability of 0.1
    let mut rrFactor = 1.0;
    if depth >= 5 {
        let rrStopProbability = 0.1;
        if random::<f64>() <= rrStopProbability {
            *clr += scene.background_color;
            return;
        }
        rrFactor = 1.0 / (1.0 - rrStopProbability);
    }

    let intersection = scene.intersect(ray);
    let (t, obj) = if let Some(obj) = intersection.object {
        (intersection.t, obj)
    } else {
        *clr += scene.background_color;
        return;
    };

    // Travel the ray to the hit point where the closest object lies and compute the surface normal there.
    let hp = ray.o + ray.d.scale(t);
    let mut N = obj.shape.normal(&hp);
    ray.o = hp;
    // Add the emission, the L_e(x,w) part of the rendering equation, but scale it with the Russian Roulette
    // probability weight.
    *clr += v3(obj.emission, obj.emission, obj.emission).scale(rrFactor);

    let mut tmp = v3(0.0, 0.0, 0.0);
    match obj.obj_type {
        // Diffuse BRDF - choose an outgoing direction with hemisphere sampling.
        ObjType::DiffuseBRDF => {
            let (rotX, rotY) = ons(&N);
            let sampledDir = hemisphere(random(), random());
            let rotatedDir = V3 {
                x: v3(rotX.x, rotY.x, N.x).dot(&sampledDir),
                y: v3(rotX.y, rotY.y, N.y).dot(&sampledDir),
                z: v3(rotX.z, rotY.z, N.z).dot(&sampledDir),
            };
            ray.d = rotatedDir; // already normalized
            let cost = ray.d.dot(&N);
            trace(ray, scene, depth + 1, &mut tmp);
            *clr += (tmp * obj.cl).scale(cost * 0.1 * rrFactor);
        }
        // Specular BRDF - this is a singularity in the rendering equation that follows
        // delta distribution, therefore we handle this case explicitly - one incoming
        // direction -> one outgoing direction, that is, the perfect reflection direction.
        ObjType::SpecularBRDF => {
            let cost = ray.d.dot(&N);
            ray.d = (ray.d - N.scale(cost * 2.0)).norm();
            trace(ray, scene, depth + 1, &mut tmp);
            *clr += tmp.scale(rrFactor);
        }
        // Glass/refractive BRDF - we use the vector version of Snell's law and Fresnel's law
        // to compute the outgoing reflection and refraction directions and probability weights.
        ObjType::GlassRefractiveBRDF => {
            let mut n = scene.refr_index;
            let mut R0 = (1.0 - n) / (1.0 + n);
            R0 = R0 * R0;
            if N.dot(&ray.d) > 0.0 {
                // we're inside the medium
                N = N.scale(-1.0);
                n = 1.0 / n;
            }
            n = 1.0 / n;
            let cost1 = (N.dot(&ray.d)) * -1.0; // cosine of theta_1
            let cost2 = 1.0 - n * n * (1.0 - cost1 * cost1); // cosine of theta_2
            let Rprob = R0 + (1.0 - R0) * f64::powf(1.0 - cost1, 5.0); // Schlick-approximation
            ray.d = if cost2 > 0.0 && random::<f64>() > Rprob {
                // refraction direction
                ((ray.d.scale(n)) + (N.scale(n * cost1 - cost2.sqrt()))).norm()
            } else {
                // reflection direction
                (ray.d + N.scale(cost1 * 2.0)).norm()
            };
            trace(ray, scene, depth + 1, &mut tmp);
            *clr += tmp.scale(1.15 * rrFactor);
        }
    }
}

fn v3(x: f64, y: f64, z: f64) -> V3 {
    V3 { x, y, z }
}

fn sphere(r: f64, c: V3) -> Shape {
    Shape::Sphere(Sphere { c, r })
}

fn plane(d: f64, n: V3) -> Shape {
    Shape::Plane(Plane { n, d })
}

fn main() -> std::io::Result<()> {
    let scene = Scene {
        width: 900,
        height: 900,
        samples_per_pixel: 256,
        refr_index: 1.5,
        fovx: FRAC_PI_4,
        background_color: v3(1.0, 1.0, 1.0).scale(4.0),
        objects: vec![
            // Middle sphere
            Obj {
                cl: v3(4.0, 8.0, 4.0),
                emission: 0.0,
                obj_type: ObjType::SpecularBRDF,
                shape: sphere(1.05, v3(-0.75, -1.45, -4.4)),
            },
            // Right sphere
            Obj {
                cl: v3(10.0, 10.0, 1.0),
                emission: 0.0,
                obj_type: ObjType::GlassRefractiveBRDF,
                shape: sphere(0.5, v3(2.0, -2.05, -3.7)),
            },
            // Left sphere
            Obj {
                cl: v3(4.0, 4.0, 12.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
                shape: sphere(0.6, v3(-1.75, -1.95, -3.1)),
            },
            // Bottom plane
            Obj {
                shape: plane(2.5, v3(0.0, 1.0, 0.0)),
                cl: v3(6.0, 6.0, 6.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
            },
            // Back plane
            Obj {
                shape: plane(5.5, v3(0.0, 0.0, 1.0)),
                cl: v3(6.0, 6.0, 6.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
            },
            // Left plane
            Obj {
                shape: plane(2.75, v3(1.0, 0.0, 0.0)),
                cl: v3(10.0, 2.0, 2.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
            },
            // Right plane
            Obj {
                shape: plane(2.75, v3(-1.0, 0.0, 0.0)),
                cl: v3(2.0, 10.0, 2.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
            },
            // Ceiling plane
            Obj {
                shape: plane(3.0, v3(0.0, -1.0, 0.0)),
                cl: v3(6.0, 6.0, 6.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
            },
            // Front plane
            Obj {
                shape: plane(0.5, v3(0.0, 0.0, -1.0)),
                cl: v3(6.0, 6.0, 6.0),
                emission: 0.0,
                obj_type: ObjType::DiffuseBRDF,
            },
            // Light
            Obj {
                shape: sphere(0.5, v3(0.0, 1.9, -3.0)),
                cl: v3(0.0, 0.0, 0.0),
                emission: 10000.0,
                obj_type: ObjType::DiffuseBRDF,
            },
        ],
    };

    let mut pix = vec![vec![v3(0.0, 0.0, 0.0); scene.width]; scene.height];

    let start = Instant::now();

    let progress = std::sync::atomic::AtomicUsize::new(0);

    pix.par_iter_mut().enumerate().for_each(|(col, row_data)| {
        let old = progress.fetch_add(1, Ordering::SeqCst);
        print!(
            "\rRendering: {}spp {:8.2}%",
            scene.samples_per_pixel,
            old as f64 / scene.height as f64 * 100.0
        );
        std::io::stdout().flush().unwrap();
        row_data.iter_mut().enumerate().for_each(|(row, data)| {
            let mut color = v3(0.0, 0.0, 0.0);
            let cam = scene.camcr(col as f64, row as f64); // construct image plane coordinates
            for _ in 0..scene.samples_per_pixel {
                let cam_2 = v3(
                    cam.x + (random::<f64>() - 0.5) / scene.width as f64, // anti-aliasing for free
                    cam.y + (random::<f64>() - 0.5) / scene.height as f64,
                    cam.z,
                );

                let o = v3(0.0, 0.0, 0.0);
                let mut ray = Ray {
                    o,
                    d: (cam_2 - o).norm(), // point from the origin to the camera plane
                };

                trace(&mut ray, &scene, 0, &mut color);
            }
            *data += color.div(scene.samples_per_pixel as f64); // write the contributions
        });
    });

    let mut file = File::create(format!("ray-{}.ppm", scene.samples_per_pixel))?;
    write!(file, "P3\n{} {}\n{}\n ", scene.width, scene.height, 255)?;
    for row in 0..scene.height {
        for col in 0..scene.width {
            let p = pix[col][row];
            write!(
                file,
                "{} {} {} ",
                min(p.x as i64, 255),
                min(p.y as i64, 255),
                min(p.z as i64, 255)
            )?;
        }
        write!(file, "\n")?;
    }

    println!("\nRender time: {}s.", start.elapsed().as_secs_f64());
    Ok(())
}
