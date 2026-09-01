{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    cargo
    clang
    clippy
    glibc.bin
    pkg-config
    rustc
    rustfmt
  ];

  shellHook = ''
    export CC="${pkgs.clang}/bin/clang"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    if [ -f /run/opengl-driver/lib/libcuda.so.1 ]; then
      export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
    fi
  '';
}
