const std = @import("std");
const testing = std.testing;
const Cuda = @import("cudaz");
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;
const Function = Cuda.Device.CudaFunction;
const Module = Cuda.Device.Module;

pub const Kernel = struct {
    ptx: [:0]const u8,
    module: Module,
    func: Function,
    device: CuDevice,

    pub fn init(alloc: std.mem.Allocator) !Kernel {
        const device = try CuDevice.default();

        const ptx = try read_source("zig-out/lib/wmma", alloc);
        // defer alloc.free(ptx);

        const module = try CuDevice.loadPtxText(ptx);
        const function = try module.getFunc("test_wmma");

        return .{ .func = function, .module = module, .ptx = ptx, .device = device };
    }

    pub fn run(self: Kernel, a: []f16, b: []f16, c: []f16, stride: usize, allocator: std.mem.Allocator) !std.ArrayList(f16) {
        const cu_slice_a = try self.device.htodCopy(f16, a);
        const cu_slice_b = try self.device.htodCopy(f16, b);
        const dest_cu_slice = try self.device.htodCopy(f16, c);
        defer cu_slice_a.free();
        defer cu_slice_b.free();
        defer dest_cu_slice.free();

        try self.func.run(.{ &cu_slice_a.device_ptr, &cu_slice_b.device_ptr, &dest_cu_slice.device_ptr, &stride }, CuLaunchConfig{ .block_dim = .{ 1024, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
        const result = try CuDevice.syncReclaim(f16, allocator, dest_cu_slice);

        return result;
    }

    pub fn deinit(self: Kernel, alloc: std.mem.Allocator) void {
        alloc.free(self.ptx);
        self.device.deinit();
    }
};

// pub fn wmmBlock(a: []f16, b: []f16, c: []f16, stride: usize, allocator: std.mem.Allocator) !std.ArrayList(f16) {
//     const device = try CuDevice.default();
//     defer device.deinit();
//
//     const cu_slice_a = try device.htodCopy(f16, a);
//     const cu_slice_b = try device.htodCopy(f16, b);
//     const dest_cu_slice = try device.htodCopy(f16, c);
//     defer cu_slice_a.free();
//     defer cu_slice_b.free();
//     defer dest_cu_slice.free();
//
//     try function.run(.{ &cu_slice_a.device_ptr, &cu_slice_b.device_ptr, &dest_cu_slice.device_ptr, &stride }, CuLaunchConfig{ .block_dim = .{ 1024, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
//     const result = try CuDevice.syncReclaim(f16, allocator, dest_cu_slice);
//
//     return result;
// }

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
