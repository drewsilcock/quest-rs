#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(test)]
mod tests {
    use super::bindings::*;

    #[test]
    fn run_circuit() {
        unsafe {
            let env = createQuESTEnv();

            let reg = createQureg(2, env);

            destroyQureg(reg, env);
            destroyQuESTEnv(env);
        }
    }
}
