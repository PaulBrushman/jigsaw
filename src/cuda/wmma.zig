const std = @import("std");
const testing = std.testing;
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;

pub fn wmmBlock(a: []f16, b: []f16, c: []f16, stride: usize, allocator: std.mem.Allocator) !std.ArrayList(f16) {
    const device = try CuDevice.default();
    defer device.deinit();

    const cu_slice_a = try device.htodCopy(f16, a);
    const cu_slice_b = try device.htodCopy(f16, b);
    const dest_cu_slice = try device.htodCopy(f16, c); //try device.alloc(f16, 256);
    defer cu_slice_a.free();
    defer cu_slice_b.free();
    defer dest_cu_slice.free();

    const ptx = try read_source("src/cuda/wmma", allocator);
    defer allocator.free(ptx);

    const module = try CuDevice.loadPtxText(ptx);
    const function = try module.getFunc("test_wmma");

    try function.run(.{ &cu_slice_a.device_ptr, &cu_slice_b.device_ptr, &dest_cu_slice.device_ptr, stride }, CuLaunchConfig{ .block_dim = .{ 1024, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
    const result = try CuDevice.syncReclaim(f16, allocator, dest_cu_slice);

    return result;
}

fn read_source(source: []const u8, alloc: std.mem.Allocator) ![:0]const u8 {
    const source_file = try std.fs.cwd().openFile(source, .{});
    defer source_file.close();
    const size = (try source_file.stat()).size;
    const buff = try alloc.alloc(u8, size + 1);
    const bytes_read = try source_file.readAll(buff);
    try std.testing.expect(bytes_read == size);
    buff[size] = 0;
    return buff[0..size :0];
}
