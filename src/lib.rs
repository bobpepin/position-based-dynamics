// Based on "Detailed Rigid Body Simulation with Extended Position Based Dynamics" (2020) by Müller et al.

extern crate wee_alloc;
extern crate nalgebra as na;
use na::{Vector3, Quaternion};

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[no_mangle]
pub extern fn add(a: u32, b: u32) -> u32 {
    return a + b;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Geometry {
    None,
    Ball { radius: f32 },
    Plane { normal: Vector3<f32> }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Position {
    x: Vector3<f32>,
    q: Quaternion<f32>
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Body {
    inv_mass: f32,
    inv_inertia: na::Matrix3<f32>
}


impl Position {
    fn new(x: f32, y: f32, z: f32) -> Position {
        return Position {
            x: Vector3::new(x, y, z),
            q: Quaternion::new(1.0, 0.0, 0.0, 0.0)
        }
    }
}

#[no_mangle]
pub extern fn allocStateVector(n: usize) -> *mut Vec<Position> {
    let init_elt = Position { 
        x: Vector3::new(0.0, 0.0, 0.0), 
        q: Quaternion::new(0.0, 0.0, 0.0, 0.0) 
    };
    let v = vec![init_elt; n];
    return Box::into_raw(Box::new(v))
}

#[no_mangle]
pub extern fn allocGeometryVector(n: usize) -> *mut Vec<Geometry> {
    let init_elt = Geometry::None;
    let v = vec![init_elt; n];
    return Box::into_raw(Box::new(v))
}

#[no_mangle]
pub extern fn allocBodyVector(n: usize) -> *mut Vec<Body> {
    let init_elt = Body {
        inv_mass: 0.0,
        inv_inertia: na::Matrix3::identity()
    };
    let v = vec![init_elt; n];
    return Box::into_raw(Box::new(v))
}

fn get_data_pointer<T>(ptr: *const Vec<T>) -> *const T {
    let v;
    unsafe {
        v = &*ptr;
    }
    return v.as_ptr();
}

#[no_mangle]
pub extern fn getStatePointer(ptr: *const Vec<Position>) -> *const Position {
    return get_data_pointer(ptr);
}

#[no_mangle]
pub extern fn getGeometryPointer(ptr: *const Vec<Geometry>) -> *const Geometry {
    return get_data_pointer(ptr);
}

#[no_mangle]
pub extern fn getBodyPointer(ptr: *const Vec<Body>) -> *const Body {
    return get_data_pointer(ptr);
}

fn detect_intersection(p1: Position, g1: Geometry, p2: Position, g2: Geometry) -> bool {
    match (g1, g2) {
        (Geometry::Ball { radius: r1 }, Geometry::Ball { radius: r2 }) 
            => (p1.x - p2.x).norm_squared() < (r1 + r2).powf(2.0),
        _ => false
    }
}

fn compute_delta(p1: Position, g1: Geometry, p2: Position, g2: Geometry) -> Position {
    match (g1, g2) {
        (Geometry::Ball { radius: r1 }, Geometry::Ball { radius: r2 }) => {
            let v = p2.x - p1.x; // This is the inverse of the paper
            let distance = v.norm();
            let delta_norm = r1 + r2 - distance;
            let delta = delta_norm * (v / distance);
            Position { x: delta, q: Quaternion::identity() }
        },
        _ => Position::new(0.0, 0.0, 0.0)
    }
}

fn compute_update(_p1: Position, b1: Body, _p2: Position, b2: Body, delta: Position) -> (Position, Position) {
    let w1 = b1.inv_mass;
    let w2 = b2.inv_mass;
    let c = delta.x.norm();
    let n = delta.x / c;
    let delta_lambda = -c / (w1 + w2);
    let p = delta_lambda * n;
    let delta_x1 = p * w1;
    let delta_x2 = -p * w2;
    let update1 = Position { x: delta_x1, q: Quaternion::from_real(0.0) };
    let update2 = Position { x: delta_x2, q: Quaternion::from_real(0.0) };
    return (update1, update2);
}

fn solve_positions(positions: &mut Vec<Position>, geometries: &Vec<Geometry>, bodies: &Vec<Body>) {
    let n = positions.len();
    for i1 in 0..n {
        let p1 = positions[i1];
        let g1 = geometries[i1];
        let b1 = bodies[i1];
        for i2 in 0..i1 {
            let p2 = positions[i2];
            let g2 = geometries[i2];
            let b2 = bodies[i2];
            if detect_intersection(p1, g1, p2, g2) {
                let delta = compute_delta(p1, g1, p2, g2);
                let (update1, update2) = compute_update(p1, b1, p2, b2, delta);
                positions[i1].x += update1.x;
                positions[i1].q += update1.q;
                positions[i2].x += update2.x;
                positions[i2].q += update2.q;
            }
        }
    }
}

#[no_mangle]
pub extern fn solvePositions(
    positions_ptr: *mut Vec<Position>,
    geometries_ptr: *const Vec<Geometry>,
    bodies_ptr: *const Vec<Body>
) {
    let positions_ref;
    let geometries_ref;
    let bodies_ref;
    unsafe {
        positions_ref = &mut *positions_ptr;
        geometries_ref = &*geometries_ptr;
        bodies_ref = &*bodies_ptr;
    }
    solve_positions(positions_ref, geometries_ref, bodies_ref);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        let x1 = Position::new(-1.0, 0.0, 0.0);
        let x2 = Position::new(1.0, 0.0, 0.0);
        let g1 = Geometry::Ball { radius: 0.5 };
        let g2 = Geometry::Ball { radius: 1.0 };
        let g3 = Geometry::Ball { radius: 1.5 };
        assert!(!detect_intersection(x1, g1, x2, g1));
        assert!(!detect_intersection(x1, g1, x2, g2));
        assert!(!detect_intersection(x1, g2, x2, g2));
        assert!(detect_intersection(x1, g2, x2, g3));
        assert!(detect_intersection(x1, g2, x1, g3));
    }
    
    #[test]
    fn test_delta() {
        let x1 = Position::new(-1.0, 0.0, 0.0);
        let x2 = Position::new(0.0, 0.0, 0.0);
        let g1 = Geometry::Ball { radius: 0.6 };
        let delta = compute_delta(x1, g1, x2, g1);
        assert!((delta.x - Vector3::new(0.2, 0.0, 0.0)).norm() < 1e-5);
    }
    
    #[test]
    fn test_update() {
        let x1 = Position::new(-1.0, 0.0, 0.0);
        let x2 = Position::new(0.0, 0.0, 0.0);
        let g1 = Geometry::Ball { radius: 0.6 };        
        let b1 = Body { inv_mass: 1.0, inv_inertia: na::Matrix3::identity() };
        let delta = compute_delta(x1, g1, x2, g1);
        let (dx1, dx2) = compute_update(x1, b1, x2, b1, delta);
        assert!(
            (dx1.x - Vector3::new(-0.1, 0.0, 0.0)).norm() < 1e-5,
            "delta = {:?}, dx1 = {:?}, dx2 = {:?}", delta, dx1, dx2
        );
        assert!((dx2.x - Vector3::new(0.1, 0.0, 0.0)).norm() < 1e-5);        
    }
    
    #[test]
    fn test_solve() {
        let x1 = Position::new(-1.0, 0.0, 0.0);
        let x2 = Position::new(0.0, 0.0, 0.0);
        let g1 = Geometry::Ball { radius: 0.6 };        
        let b1 = Body { inv_mass: 1.0, inv_inertia: na::Matrix3::identity() };
        let mut positions = vec![x1, x2];
        let geometries = vec![g1, g1];
        let bodies = vec![b1, b1];
        solve_positions(&mut positions, &geometries, &bodies);
        assert!((positions[0].x - Vector3::new(-1.1, 0.0, 0.0)).norm() < 1e-5);
        assert!((positions[1].x - Vector3::new(0.1, 0.0, 0.0)).norm() < 1e-5);
    }
}