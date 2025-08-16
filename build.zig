const std = @import("std");

pub fn build(b: *std.Build) void {
    const nvrtc = b.addExecutable(.{ .name = "cuda_compile", .root_source_file = b.path("tools/cuda_compile.zig"), .target = b.graph.host });
    const cudaz_dep = b.dependency("cudaz", .{ .CUDA_PATH = @as([]const u8, std.posix.getenv("CUDA_PATH").?) });
    const cudaz_module = cudaz_dep.module("cudaz");
    nvrtc.root_module.addImport("cudaz", cudaz_module);
    nvrtc.linkLibC();
    nvrtc.linkSystemLibrary("cuda");
    nvrtc.linkSystemLibrary("nvrtc");

    const utils = b.createModule(.{ .root_source_file = b.path("utils/utils.zig") });

    const compile_cuda = b.addRunArtifact(nvrtc);
    compile_cuda.addArg("src/cuda/wmma.cu");
    compile_cuda.addArg("zig-out/lib/wmma");
    b.getInstallStep().dependOn(&compile_cuda.step);

    const wmma = b.createModule(.{ .root_source_file = b.path("src/cuda/wmma.zig") });
    wmma.addImport("cudaz", cudaz_module);
    wmma.addImport("utils", utils);

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/zig/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib_mod.addImport("wmma", wmma);
    lib_mod.addImport("utils", utils);

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "wmmatest",
        .root_module = lib_mod,
    });
    lib.linkLibC();
    lib.linkSystemLibrary("cuda");
    lib.linkSystemLibrary("nvrtc");

    b.installArtifact(lib);
    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(b.getInstallStep());
    test_step.dependOn(&run_lib_unit_tests.step);
}
