const std = @import("std");
const testing = std.testing;
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const args = try std.process.argsAlloc(allocator);
    if (args.len != 3) std.process.exit(2);
    const source = args[1];
    const target = args[2];

    const device = try CuDevice.default();
    defer device.deinit();
    // std.debug.print("Cuda device is setup\n", .{});
    const env_vars = [_][]const u8{ "CUDA_PATH", "NIX_LDFLAGS", "EXTRA_LDFLAGS", "EXTRA_CCFLAGS", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES" }; // "LDFLAGS", "STDDEV_PATH", "CUDA_DISABLE_PTX_JIT"
    var env_values: [env_vars.len + 1][]const u8 = .{undefined} ** (env_vars.len + 1);
    inline for (env_vars, 0..) |_var, i|
        env_values[i] = try std.process.getEnvVarOwned(allocator, _var);
    env_values[env_values.len - 1] =
        try std.mem.concat(allocator, u8, &.{ env_values[0], "/include" });
    defer inline for (env_values) |value| allocator.free(value);

    var d = [_][]const u8{"compute_86"};
    var m = [_][]const u8{"__x86_64__"};

    const wmma_kernel = try read_source(source, allocator);
    defer allocator.free(wmma_kernel);

    const ptx = try CuCompile.cudaText(wmma_kernel, .{ .include_paths = &env_values, .arch = &d, .macro = &m, .rdc = true }, allocator);
    defer allocator.free(ptx);

    const target_file = try std.fs.cwd().createFile(target, .{});
    defer target_file.close();
    try target_file.writeAll(ptx);
}

fn read_source(source: [:0]u8, alloc: Allocator) ![:0]const u8 {
    const source_file = try std.fs.cwd().openFile(source, .{});
    defer source_file.close();
    const size = (try source_file.stat()).size;
    const buff = try alloc.alloc(u8, size + 1);
    const bytes_read = try source_file.readAll(buff);
    try expect(bytes_read == size);
    buff[size] = 0;
    return buff[0..size :0];
}
