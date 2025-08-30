const std = @import("std");
const testing = std.testing;
const Cuda = @import("cudaz");
const Precision = @import("utils").Precision;
const CuDevice = Cuda.Device;
const CuCompile = Cuda.Compile;
const CuLaunchConfig = Cuda.LaunchConfig;
const Function = Cuda.Function;
const Module = Cuda.Module;
// const print = std.debug.print;
// const cudaMalloc = @cImport("cuda_runtime.h").cudaMalloc;
const prec_num: usize = @typeInfo(Precision).@"enum".fields.len;

pub const Kernel = struct {
    ptx: [:0]const u8,
    module: Module,
    func: [prec_num]Function,
    device: CuDevice,

    pub fn init(alloc: std.mem.Allocator) !Kernel {
        const device = try CuDevice.default();

        const ptx = try read_source("zig-out/lib/wmma", alloc);

        const module = try CuDevice.loadPtxText(ptx);
        var function = [_]Function{undefined} ** prec_num;
        inline for (@typeInfo(Precision).@"enum".fields, 0..) |name, i|
            function[i] = try module.getFunc("wmma_" ++ name.name);
        return .{ .func = function, .module = module, .ptx = ptx, .device = device };
    }

    pub fn run(self: Kernel, a: []f16, b: []f16, c: []f16, precision: Precision, stride: usize, allocator: std.mem.Allocator) !std.ArrayList(f16) {
        const cu_slice_a = try self.device.htodCopy(f16, a);
        const cu_slice_b = try self.device.htodCopy(f16, b);
        const dest_cu_slice = try self.device.htodCopy(f16, c);
        const b_a = try self.device.htodCopy(u8, &([_]u8{0} ** 256));
        const b_b = try self.device.htodCopy(u8, &([_]u8{0} ** 256));
        const b_c = try self.device.htodCopy(i32, &([_]i32{0} ** 256));
        defer b_a.free();
        defer b_b.free();
        defer b_c.free();
        defer cu_slice_a.free();
        defer cu_slice_b.free();
        defer dest_cu_slice.free();

        try self.func[@intFromEnum(precision)].run(.{ &cu_slice_a.device_ptr, &cu_slice_b.device_ptr, &dest_cu_slice.device_ptr, &b_a.device_ptr, &b_b.device_ptr, &b_c.device_ptr, &stride }, CuLaunchConfig{ .block_dim = .{ 1024, 1, 1 }, .grid_dim = .{ 1, 1, 1 }, .shared_mem_bytes = 0 });
        const result = try CuDevice.syncReclaim(f16, allocator, dest_cu_slice);

        return result;
    }

    pub fn deinit(self: Kernel, alloc: std.mem.Allocator) void {
        alloc.free(self.ptx);
        self.device.deinit();
    }
};

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
