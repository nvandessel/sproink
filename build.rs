fn main() {
    // cbindgen 0.27 doesn't support Rust 2024 edition's #[unsafe(no_mangle)]
    // and `unsafe extern "C"` syntax. The sproink.h header is maintained manually.
    // When cbindgen gains 2024 edition support, re-enable auto-generation:
    //
    // let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    // cbindgen::Builder::new()
    //     .with_crate(&crate_dir)
    //     .with_config(cbindgen::Config::from_file("cbindgen.toml").unwrap())
    //     .generate()
    //     .expect("Unable to generate C bindings")
    //     .write_to_file("sproink.h");
}
