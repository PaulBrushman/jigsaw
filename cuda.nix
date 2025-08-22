{ pkgs ? import <nixpkgs> {
  config.allowUnfree = true;
  config.cudaSupport = true;
  config.cudaVersion = 12;
  } }:
pkgs.mkShell {
  name = "cuda-env-shell";
  buildInputs = with pkgs; [
    git gitRepo gnupg 
    # autoconf 
    curl
    # procps gnumake util-linux m4 gperf unzip
    cudatoolkit # cudaPackages.cuda_nvcc 
    glibc_multi
    # libGLU libGL
    xorg.libXi xorg.libXmu 
    # freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    linuxPackages.nvidia_x11
    zlib ncurses5 binutils
    # stdenv.cc stdenv.cc.cc
    cudaPackages.cuda_cudart
    # cudaPackages.
  ];
  # flake.nix, run with `nix develop`
  shellHook = ''
    
        # export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH"
        export CUDA_PATH=${pkgs.cudatoolkit}
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        export CMAKE_PREFIX_PATH="${pkgs.fmt.dev}:$CMAKE_PREFIX_PATH"
        export PKG_CONFIG_PATH="${pkgs.fmt.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"
    # export CUDA_DISABLE_PTX_JIT=0;
    # export CUDA_PATH=${pkgs.cudatoolkit};
    export CUDA_VISIBLE_DEVICES=1;
    # export EXTRA_NVCCFLAGS=--cudart=shared
    # export LD_LIBRARY_PATH=${pkgs.cudatoolkit}/include:${pkgs.cudatoolkit}/lib:/run/opengl-driver/lib:${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
    # export LDFLAGS="/nix/store/r25srliigrrv5q3n7y8ms6z10spvjcd9-glibc-2.40-66-dev/include"
    # export STDDEV_PATH=${pkgs.stdenv.cc.cc}/lib/gcc/x86_64-unknown-linux-gnu/14.3.0/include;
    # -L/lib -L/nix/store/zwx8hvsrzwk87v84q60a0pxv256jmpjz-zig-0.13.0/lib/zig/libcxx/include 
    # r25srliigrrv5q3n7y8ms6z10spvjcd9-glibc-2.40-66-dev/include/stdlib.h

    # export EXTRA_LDFLAGS=${pkgs.linuxPackages.nvidia_x11}/lib
    # export EXTRA_CCFLAGS="/usr/include"
    # export EXTRA_LDFLAGS="-L/lib/ -L${pkgs.linuxPackages.nvidia_x11}/lib -L${pkgs.cudatoolkit}/include"
    # export EXTRA_CCFLAGS="-I/usr/include -I${pkgs.cudatoolkit}/include"
    # export LDFLAGS="-L/lib/ -L${pkgs.linuxPackages.nvidia_x11}/lib -L${pkgs.cudatoolkit}/include"
    # export CCFLAGS="-I/usr/include -I${pkgs.cudatoolkit}/include"
    # export NIX_LDFLAGS="$NIX_LDFLAGS -L${pkgs.cudatoolkit}/lib" # -L/nix/store/7r9q014pymaibifx57rz1cd8xldcmlaj-cudatoolkit-12.8.0/targets/x86_64-linux/lib
  '';
}
