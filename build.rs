use bindgen;
use cmake;

use std::env;
use std::path::PathBuf;

fn main() {
    let dst = cmake::Config::new("QuEST").no_build_target(true).build();
    println!("cargo:rustc-link-search=native={}/build/QuEST", dst.display());
    println!("cargo:rustc-link-lib=dylib=QuEST");

    let bindings = bindgen::Builder::default()
        .header("QuEST/QuEST/include/QuEST.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
