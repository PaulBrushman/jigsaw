{ pkgs ? import <nixpkgs> {
  config.allowUnfree = true;
  config.cudaSupport = true;
  config.cudaVersion = 12;
  } }:
# pkgs.pkgsi686Linux.mkShell {
pkgs.mkShell {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    git gitRepo gnupg autoconf curl
    procps gnumake util-linux m4 gperf unzip
    cudatoolkit # cudaPackages.cuda_nvcc 
    glibc_multi libGLU libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    linuxPackages.nvidia_x11
    zlib ncurses5 stdenv.cc binutils
  ];
  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
    export NIX_LDFLAGS="$NIX_LDFLAGS -L/nix/store/7r9q014pymaibifx57rz1cd8xldcmlaj-cudatoolkit-12.8.0/targets/x86_64-linux/lib"
  '';
}
